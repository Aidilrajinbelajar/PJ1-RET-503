#!/usr/bin/env python3
"""Bagi dataset berlabel (format YOLO) menjadi train/val/test, buat data.yaml dan statistik.

Masukan: folder berisi gambar (.jpg/.jpeg/.png) dan file label .txt bernama sama (hasil LabelImg, CVAT,
atau Roboflow dalam format "YOLO"). Boleh satu folder campur, atau folder images/ dan labels/ terpisah.
Satu baris label: <id_kelas> <x_tengah> <y_tengah> <lebar> <tinggi>  (semua 0-1).

Contoh:
    python make_dataset.py --src dataset_raw --out dataset --names mask glasses gloves
    python make_dataset.py --src dataset_raw --out dataset --names mask glasses gloves --empty-is-negative

Foto TANPA file label dilewati (diberi peringatan), kecuali kamu pakai --empty-is-negative: foto itu dianggap
"tidak ada APD sama sekali" (contoh negatif). LabelImg tidak membuat file label untuk foto tanpa objek.
"""
import argparse
import os
import random
import shutil
from collections import Counter
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}


def find_label(img: Path):
    """Cari file label untuk gambar: sebelah gambar, atau di folder labels/ yang sejajar dengan images/."""
    cands = [img.with_suffix(".txt")]
    parts = list(img.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        cands.append(Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt"))
    for c in cands:
        if c.exists():
            return c
    return None


def read_label(path: Path, n_classes: int):
    """Baca dan validasi file label. Mengembalikan list (kelas, x, y, w, h). ValueError kalau ada yang salah."""
    rows = []
    for ln, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        where = f"{path}:{ln}"
        if len(parts) != 5:
            raise ValueError(f"{where}: harus 5 angka per baris (format kotak), dapat {len(parts)}. "
                             "File dari tool segmentasi/poligon tidak didukung.")
        try:
            cls = int(parts[0])
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            raise ValueError(f"{where}: ada nilai bukan angka: {line!r}")
        if not 0 <= cls < n_classes:
            raise ValueError(f"{where}: id kelas {cls} di luar rentang 0..{n_classes - 1}")
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"{where}: koordinat harus 0-1 dan lebar/tinggi > 0: {line!r}")
        rows.append((cls, x, y, w, h))
    return rows


def collect(src: Path, out: Path, n_classes: int, empty_is_negative: bool):
    items, skipped, errors, names_seen = [], [], [], set()
    for img in sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT):
        if out in img.parents:
            continue
        lbl = find_label(img)
        if lbl is None:
            if empty_is_negative:
                items.append((img, []))
            else:
                skipped.append(img)
            continue
        try:
            items.append((img, read_label(lbl, n_classes)))
        except ValueError as e:
            errors.append(str(e))
    stems = Counter(i.stem for i, _ in items)
    dup = [s for s, c in stems.items() if c > 1]
    if dup:
        errors.append(f"nama file kembar (beda folder, nama sama): {dup[:5]} ... ganti nama salah satunya")
    return items, skipped, errors


def split(items, ratios, seed):
    rnd = random.Random(seed)
    items = items[:]
    rnd.shuffle(items)
    total = sum(ratios)
    n = len(items)
    n_train = int(round(n * ratios[0] / total))
    n_val = int(round(n * ratios[1] / total))
    if n >= 3:                       # val dan test minimal 1 gambar bila diminta
        n_val = max(1, n_val)
        if ratios[2] > 0:
            n_val = min(n_val, n - n_train - 1) if n - n_train - 1 >= 1 else n_val
    parts = {"train": items[:n_train], "val": items[n_train:n_train + n_val], "test": items[n_train + n_val:]}
    return parts


def split_chrono(items, ratios):
    """Bagi per kondisi (tag di nama file) menurut urutan waktu: awal = train, tengah = val, akhir = test.
    Foto yang diambil beruntun sangat mirip; dibagi acak, foto 'kembar' bisa bocor ke val/test dan hasil
    evaluasi jadi terlalu bagus. Dengan blok waktu, kebocoran hanya mungkin di batas blok."""
    import re
    groups = {}
    for it in items:
        m = re.match(r"^(.*?)_\d{8}_\d{6}_", it[0].name)
        groups.setdefault(m.group(1) if m else "_", []).append(it)
    total = sum(ratios)
    parts = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda it: it[0].name)
        n = len(g)
        n_val = int(round(n * ratios[1] / total))
        n_test = int(round(n * ratios[2] / total))
        if n >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n_test) if ratios[2] > 0 else 0
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train, n_val, n_test = n, 0, 0
        parts["train"] += g[:n_train]
        parts["val"] += g[n_train:n_train + n_val]
        parts["test"] += g[n_train + n_val:]
    return parts


def main():
    p = argparse.ArgumentParser(description="Bagi dataset YOLO jadi train/val/test + data.yaml")
    p.add_argument("--src", required=True, help="folder gambar + label (hasil labeling)")
    p.add_argument("--out", default="dataset", help="folder hasil")
    p.add_argument("--names", nargs="+", required=True, help="nama kelas sesuai urutan id, contoh: mask glasses gloves")
    p.add_argument("--ratio", nargs=3, type=float, default=[0.7, 0.2, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42, help="supaya pembagian bisa diulang persis sama")
    p.add_argument("--random-split", action="store_true",
                   help="bagi acak (default: berurutan per kondisi menurut waktu, lebih jujur untuk foto beruntun)")
    p.add_argument("--public", help="folder hasil remap_public.py; dipakai hanya untuk TRAIN, sehingga val/test "
                   "tetap dari fotomu sendiri")
    p.add_argument("--empty-is-negative", action="store_true", help="foto tanpa file label dianggap contoh negatif")
    args = p.parse_args()

    src, out = Path(args.src).resolve(), Path(args.out).resolve()
    items, skipped, errors = collect(src, out, len(args.names), args.empty_is_negative)
    if errors:
        print("DITEMUKAN MASALAH, perbaiki dulu:")
        for e in errors[:20]:
            print("  -", e)
        if len(errors) > 20:
            print(f"  ... dan {len(errors) - 20} lainnya")
        raise SystemExit(1)
    if skipped:
        print(f"PERINGATAN: {len(skipped)} foto dilewati karena tidak ada file label "
              "(pakai --empty-is-negative kalau memang itu foto tanpa APD).")
    if len(items) < 3:
        raise SystemExit(f"Hanya {len(items)} foto berlabel; butuh minimal 3.")
    pub_items = []
    if args.public:
        pub_dir = Path(args.public).resolve()
        pub_items, _, pub_err = collect(pub_dir, out, len(args.names), True)
        if pub_err:
            raise SystemExit("Masalah di dataset publik: " + pub_err[0])
        print(f"Dataset publik: {len(pub_items)} foto ditambahkan ke train.")

    if out.exists():
        shutil.rmtree(out)
    parts = split(items, args.ratio, args.seed) if args.random_split else split_chrono(items, args.ratio)
    parts["train"] = parts["train"] + pub_items
    stats = {}
    for name, lst in parts.items():
        if not lst:
            continue
        (out / "images" / name).mkdir(parents=True)
        (out / "labels" / name).mkdir(parents=True)
        inst = Counter()
        neg = 0
        for img, rows in lst:
            shutil.copy2(img, out / "images" / name / img.name)
            (out / "labels" / name / (img.stem + ".txt")).write_text(
                "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rows))
            inst.update(c for c, *_ in rows)
            neg += not rows
        stats[name] = (len(lst), inst, neg)

    yaml = ["# dibuat oleh make_dataset.py (path dihitung dari lokasi file ini)"]
    yaml += [f"train: images/train", f"val: images/val"]
    if "test" in stats:
        yaml.append("test: images/test")
    yaml += [f"nc: {len(args.names)}", "names:"] + [f"  {i}: {n}" for i, n in enumerate(args.names)]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")

    lines = [f"Dataset: {out}", f"Total foto: {len(items) + len(pub_items)} (milikmu {len(items)}, publik {len(pub_items)})   (seed {args.seed})", ""]
    head = f"{'split':<7}{'foto':>6}{'negatif':>9}" + "".join(f"{n:>10}" for n in args.names)
    lines += [head, "-" * len(head)]
    for name, (n, inst, neg) in stats.items():
        lines.append(f"{name:<7}{n:>6}{neg:>9}" + "".join(f"{inst[i]:>10}" for i in range(len(args.names))))
    total_inst = Counter()
    for _, inst, _ in stats.values():
        total_inst.update(inst)
    lines.append(f"{'total':<7}{len(items) + len(pub_items):>6}{sum(v[2] for v in stats.values()):>9}"
                 + "".join(f"{total_inst[i]:>10}" for i in range(len(args.names))))
    warn = []
    for i, n in enumerate(args.names):
        if total_inst[i] == 0:
            warn.append(f"kelas '{n}' tidak punya satu pun kotak label")
        for split_name in ("val", "test"):
            if split_name in stats and stats[split_name][1][i] == 0:
                warn.append(f"kelas '{n}' tidak ada di split {split_name} (tambah foto atau ganti --seed)")
    if "val" not in stats:
        warn.append("tidak ada split val; tambah foto")
    if warn:
        lines += ["", "PERINGATAN:"] + [f"  - {w}" for w in warn]
    report = "\n".join(lines)
    (out / "dataset_stats.txt").write_text(report + "\n")
    print(report)
    print(f"\nSiap dipakai: python train.py --data {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
