#!/usr/bin/env python3
"""Ambil dataset PPE publik (ekspor YOLOv8 dari Roboflow, mis. 'PPE Detection_DATA.v3-ppe_dataset_6-class')
dan sisakan hanya kelas mask, glasses (goggles), gloves dengan id 0, 1, 2.

Kelas lain (vest, helmet, safety_shoe, dst.) dibuang dari label. Foto yang tidak punya APD kita
sisa dipertahankan sebagai contoh negatif, dibatasi supaya tidak mendominasi.

Contoh:
    mkdir public_raw && cd public_raw && unzip "../PPE Detection_DATA.v3-ppe_dataset_6-class.yolov8.zip" && cd ..
    python remap_public.py --src public_raw --out public_remapped
    python make_dataset.py --src dataset_raw --public public_remapped --out dataset --names mask glasses gloves

Hanya membaca gambar dan teks. File .pt/.pkl di dalam zip tidak dibuka.
"""
import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

from ppe_logic import ITEMS, build_class_map

IMG_EXT = {".jpg", ".jpeg", ".png"}


def read_names(src: Path):
    ys = sorted(src.rglob("data.yaml"), key=lambda p: len(p.parts))
    if not ys:
        raise SystemExit("data.yaml tidak ditemukan di --src (apakah zip sudah di-extract?).")
    text = ys[0].read_text()
    m = re.search(r"^names:\s*(\[.*?\])\s*$", text, re.M)
    if m:                                    # names: ['a', 'b']
        names = [s.strip().strip("'\"") for s in m.group(1).strip("[]").split(",") if s.strip()]
        return dict(enumerate(names))
    names = {}
    block = re.search(r"^names:\s*\n((?:[ \t]+.*\n?)+)", text, re.M)
    if block:                                # names:\n  0: a
        for line in block.group(1).splitlines():
            k, _, v = line.strip().partition(":")
            if k.isdigit():
                names[int(k)] = v.strip().strip("'\"")
    if not names:
        raise SystemExit(f"Tidak bisa membaca 'names' dari {ys[0]}")
    return names


def main():
    p = argparse.ArgumentParser(description="Sisakan kelas mask/glasses/gloves dari dataset PPE publik")
    p.add_argument("--src", required=True, help="folder hasil extract zip (berisi data.yaml, train/, valid/, test/)")
    p.add_argument("--out", default="public_remapped")
    p.add_argument("--max-negative-ratio", type=float, default=0.2,
                   help="foto tanpa APD kita maksimal sebesar rasio ini terhadap foto positif")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-splits", action="store_true",
                   help="pakai pembagian train/valid/test bawaan dataset dan langsung tulis dataset siap latih "
                        "(images/, labels/, data.yaml). Disarankan kalau hanya memakai dataset publik.")
    args = p.parse_args()

    src, out = Path(args.src), Path(args.out)
    names = read_names(src)
    cmap = build_class_map(names)   # id -> mask/glasses/gloves
    new_id = {i: ITEMS.index(k) for i, k in cmap.items() if k in ITEMS}
    print("Pemetaan kelas:")
    for i, n in names.items():
        print(f"  {i} {n:<14} -> {ITEMS[new_id[i]] if i in new_id else '(dibuang)'}")
    if len(new_id) < 3:
        print("PERINGATAN: tidak semua dari mask/glasses/gloves ketemu di dataset ini.")

    def split_of(img):
        for part in img.relative_to(src).parts:
            low = part.lower()
            if low == "train":
                return "train"
            if low in ("valid", "val", "validation"):
                return "val"
            if low == "test":
                return "test"
        return "train"

    import random
    rnd = random.Random(args.seed)
    groups = {}                         # split -> ([positif], [negatif])
    for img in sorted(q for q in src.rglob("*") if q.suffix.lower() in IMG_EXT):
        lbl = img.with_suffix(".txt")
        parts = list(img.parts)
        if not lbl.exists() and "images" in parts:
            i = len(parts) - 1 - parts[::-1].index("images")
            lbl = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
        if not lbl.exists():
            continue
        rows = []
        for line in lbl.read_text().splitlines():
            f = line.split()
            if len(f) != 5:
                continue            # baris poligon/segmentasi dilewati
            try:
                c = int(f[0])
            except ValueError:
                continue
            if c in new_id:
                rows.append(f"{new_id[c]} {' '.join(f[1:])}")
        key = split_of(img) if args.keep_splits else "all"
        pos, neg = groups.setdefault(key, ([], []))
        (pos if rows else neg).append((img, rows))

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    count, n_pos, n_neg, n = Counter(), 0, 0, 0
    per_split = {}
    for key, (pos, neg) in groups.items():
        rnd.shuffle(neg)
        neg = neg[: int(len(pos) * args.max_negative_ratio)]
        n_pos += len(pos)
        n_neg += len(neg)
        sc = Counter()
        for img, rows in pos + neg:
            stem = f"pub_{n:05d}"
            n += 1
            if args.keep_splits:
                (out / "images" / key).mkdir(parents=True, exist_ok=True)
                (out / "labels" / key).mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, out / "images" / key / (stem + img.suffix.lower()))
                (out / "labels" / key / (stem + ".txt")).write_text("".join(r + "\n" for r in rows))
            else:
                shutil.copy2(img, out / (stem + img.suffix.lower()))
                (out / (stem + ".txt")).write_text("".join(r + "\n" for r in rows))
            sc.update(int(r.split()[0]) for r in rows)
        count.update(sc)
        per_split[key] = (len(pos) + len(neg), sc)

    if args.keep_splits:
        yaml = ["train: images/train", "val: images/val"]
        if "test" in groups:
            yaml.append("test: images/test")
        yaml += ["nc: 3", "names:"] + [f"  {i}: {k}" for i, k in enumerate(ITEMS)]
        (out / "data.yaml").write_text("\n".join(yaml) + "\n")
    print(f"\nDisimpan di '{out}': {n_pos} foto positif + {n_neg} negatif")
    for key, (tot, sc) in per_split.items():
        if args.keep_splits:
            print(f"  {key:<6} {tot:>5} foto  " + "  ".join(f"{k}={sc[i]}" for i, k in enumerate(ITEMS)))
    for i, k in enumerate(ITEMS):
        print(f"  {k:<8} {count[i]} kotak")
    if args.keep_splits:
        print(f"\nSiap dipakai: python train.py --data {out / 'data.yaml'}")

if __name__ == "__main__":
    main()
