#!/usr/bin/env python3
"""Latih dan bandingkan tiga eksperimen YOLO (praktik terpadu RET503, tahap 3-5).

    E1_pretrained : yolo26n.pt   (transfer learning dari COCO)         -> baseline
    E2_scratch    : yolo26n.yaml (arsitektur sama, bobot acak)          -> bukti manfaat transfer learning
    E3_tuning     : yolo26n.pt + SATU perubahan (default imgsz=640)     -> hasil satu langkah tuning

Data, epoch, seed, dan batch sama untuk ketiganya. Model dipilih dari mAP50-95 di data VAL tertinggi,
lalu data TEST dipakai SEKALI saja untuk model terpilih.

Contoh:
    python latih_bandingkan.py --nim 4222411014 --nama "Yasin Aprian" --data /content/data_gabungan.yaml
    python latih_bandingkan.py --nim 4222411014 --nama "Yasin Aprian" --data data.yaml --ubah imgsz=640
    python latih_bandingkan.py ... --epochs 1 --fraction 0.05 --out uji_cepat      # uji cepat saja

Keluaran: <out>/E1_pretrained/, <out>/E2_scratch/, <out>/E3_tuning/ (weights, results.csv, results.png, ...),
dan hasil_praktik_<NIM>.txt / .csv di folder kerja. Eksperimen yang sudah selesai dilewati kalau program
dijalankan ulang (checkpoint per eksperimen), jadi aman kalau sesi Colab terputus di tengah.
"""
import argparse
import csv
import hashlib
import json
import platform
import re
import time
from pathlib import Path

EXPERIMENTS = ("E1_pretrained", "E2_scratch", "E3_tuning")


def parse_change(text):
    """'imgsz=640' -> ('imgsz', 640). Nilai diubah ke int/float/bool bila bisa."""
    if "=" not in text:
        raise SystemExit(f"--ubah harus berbentuk kunci=nilai, dapat: {text!r}")
    key, _, raw = text.partition("=")
    key, raw = key.strip(), raw.strip()
    low = raw.lower()
    if low in ("true", "false"):
        return key, low == "true"
    for cast in (int, float):
        try:
            return key, cast(raw)
        except ValueError:
            pass
    return key, raw


def read_yaml_dirs(yaml_path):
    """Ambil daftar folder gambar train/val/test dari data.yaml sederhana (nilai tunggal atau daftar)."""
    import yaml
    cfg = yaml.safe_load(Path(yaml_path).read_text())
    base = Path(cfg.get("path", Path(yaml_path).resolve().parent))
    if not base.is_absolute():
        base = (Path(yaml_path).resolve().parent / base).resolve()
    out = {}
    for split in ("train", "val", "test"):
        v = cfg.get(split)
        if not v:
            continue
        items = v if isinstance(v, list) else [v]
        out[split] = [Path(i) if Path(i).is_absolute() else base / i for i in items]
    return out, cfg


def dataset_fingerprint(yaml_path):
    """Sidik jari dataset: hash isi file label (nama + isi) dari seluruh split. Folder yang sama (daftar berulang)
    dihitung sekali. Dua dataset dengan label berbeda hampir pasti punya sidik jari berbeda."""
    dirs, _ = read_yaml_dirs(yaml_path)
    seen, h, n_img, n_lbl = set(), hashlib.sha1(), 0, 0
    for split in sorted(dirs):
        for d in dirs[split]:
            key = (split, str(d))
            if key in seen:
                continue
            seen.add(key)
            lbl_dir = Path(str(d).replace("/images", "/labels").replace("\\images", "\\labels"))
            imgs = [p for p in sorted(d.glob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png")] if d.exists() else []
            n_img += len(imgs)
            for p in sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []:
                h.update(split.encode() + p.name.encode() + p.read_bytes())
                n_lbl += 1
    return h.hexdigest()[:12], n_img, n_lbl


def metrics_row(m, names):
    """Ambil angka ringkas dari objek hasil model.val()."""
    box = m.box
    per_class = {}
    for pos, ci in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(pos)
        per_class[names[int(ci)]] = {"P": float(p), "R": float(r), "mAP50": float(ap50), "mAP50-95": float(ap)}
    return {
        "mAP50": float(box.map50), "mAP50-95": float(box.map),
        "P": float(box.mp), "R": float(box.mr),
        "ms": float(m.speed.get("inference", 0.0)),
        "per_class": per_class,
    }


def main():
    ap = argparse.ArgumentParser(description="E1 pretrained vs E2 scratch vs E3 tuning (YOLO)")
    ap.add_argument("--nim", required=True)
    ap.add_argument("--nama", required=True)
    ap.add_argument("--data", required=True, help="data.yaml dataset projek")
    ap.add_argument("--out", default="hasil_praktik", help="folder keluaran eksperimen")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=416, help="ukuran input E1 dan E2 (baseline)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ubah", default="imgsz=640", help="satu perubahan untuk E3, bentuk kunci=nilai")
    ap.add_argument("--model", default="yolo26n", help="nama dasar model (yolo26n -> yolo26n.pt dan yolo26n.yaml)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fraction", type=float, default=1.0, help="pakai sebagian data train (untuk uji cepat)")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    from ultralytics import YOLO
    import ultralytics

    key, val = parse_change(args.ubah)
    out = Path(args.out).resolve()   # path absolut: kalau relatif, Ultralytics menaruhnya di runs/detect/<out>
    out.mkdir(parents=True, exist_ok=True)
    fp, n_img, n_lbl = dataset_fingerprint(args.data)
    _, cfg = read_yaml_dirs(args.data)
    if not cfg.get("test"):
        raise SystemExit(f"{args.data} belum punya baris 'test:'. Tambahkan folder gambar test (data test dipakai sekali di akhir).")
    names = cfg["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}

    common = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, seed=args.seed,
                  deterministic=True, project=str(out), exist_ok=True, plots=True, workers=args.workers,
                  fraction=args.fraction, verbose=False)
    if args.device is not None:
        common["device"] = args.device
    plan = {
        "E1_pretrained": (f"{args.model}.pt", {}, "-"),
        "E2_scratch": (f"{args.model}.yaml", {"pretrained": False}, "-"),
        "E3_tuning": (f"{args.model}.pt", {key: val}, f"{key}={val}"),
    }

    rows = {}
    for name in EXPERIMENTS:
        model_file, extra, change = plan[name]
        run_dir = out / name
        best = run_dir / "weights" / "best.pt"
        cache = run_dir / "ringkasan.json"
        imgsz = extra.get("imgsz", args.imgsz)
        if best.exists() and cache.exists():
            print(f"\n=== {name}: sudah selesai, dipakai ulang ({cache})")
            rows[name] = json.loads(cache.read_text())
            continue
        print(f"\n=== {name}: model={model_file} perubahan={change}")
        t0 = time.time()
        YOLO(model_file).train(name=name, **{**common, **extra})
        minutes = (time.time() - t0) / 60
        m = YOLO(str(best)).val(data=args.data, split="val", imgsz=imgsz, batch=args.batch,
                                project=str(out), name=f"{name}_val", exist_ok=True, verbose=False, plots=False)
        row = metrics_row(m, names)
        row.update({"change": change, "minutes": minutes, "imgsz": imgsz, "model": model_file})
        cache.write_text(json.dumps(row, indent=1))
        rows[name] = row

    chosen = max(rows, key=lambda k: rows[k]["mAP50-95"])
    c_imgsz = rows[chosen]["imgsz"]
    best_pt = out / chosen / "weights" / "best.pt"
    test_cache = out / "hasil_test.json"
    if test_cache.exists():
        test = json.loads(test_cache.read_text())
        if test.get("model_terpilih") != chosen:
            raise SystemExit(f"hasil_test.json sudah ada untuk {test.get('model_terpilih')} padahal terpilih {chosen}. "
                             "Data test hanya boleh dipakai sekali: hapus folder keluaran bila ingin mengulang dari awal.")
    else:
        print(f"\n=== TEST (sekali saja) untuk {chosen}")
        m = YOLO(str(best_pt)).val(data=args.data, split="test", imgsz=c_imgsz, batch=args.batch,
                                   project=str(out), name="test_terpilih", exist_ok=True, verbose=False)
        test = metrics_row(m, names)
        test["model_terpilih"] = chosen
        test_cache.write_text(json.dumps(test, indent=1))

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    host = f"{platform.node()} | {platform.platform()} | ultralytics {ultralytics.__version__}"
    nim_safe = re.sub(r"[^0-9A-Za-z_-]", "", args.nim)
    lines = [
        f"NIM / Nama    : {args.nim} / {args.nama}",
        f"Waktu         : {stamp}",
        f"Komputer      : {host}",
        f"Dataset       : {args.data}  (sidik jari label: {fp}, {n_img} gambar, {n_lbl} file label)",
        f"Pengaturan    : epochs={args.epochs} imgsz(E1,E2)={args.imgsz} batch={args.batch} seed={args.seed} E3 ubah {key}={val}",
        "",
        f"{'Eksperimen':<15}{'Perubahan':<12}{'mAP50':>7}{'mAP50-95':>10}{'P':>7}{'R':>7}{'ms':>7}{'menit':>7}",
    ]
    for n in EXPERIMENTS:
        r = rows[n]
        lines.append(f"{n:<15}{r['change']:<12}{r['mAP50']:>7.3f}{r['mAP50-95']:>10.3f}{r['P']:>7.3f}{r['R']:>7.3f}"
                     f"{r['ms']:>7.1f}{r['minutes']:>7.1f}")
    lines += [
        "",
        f"Model terpilih (mAP50-95 val tertinggi): {chosen}",
        f"Hasil di data TEST (sekali saja)       : mAP50={test['mAP50']:.3f} mAP50-95={test['mAP50-95']:.3f} "
        f"P={test['P']:.3f} R={test['R']:.3f} ({test['ms']:.1f} ms/gambar)",
        "Per kelas di data TEST:",
    ]
    for cls, v in test["per_class"].items():
        lines.append(f"  {cls:<10} P={v['P']:.3f} R={v['R']:.3f} mAP50={v['mAP50']:.3f} mAP50-95={v['mAP50-95']:.3f}")
    missing = [n for n in names.values() if n not in test["per_class"]]
    if missing:
        lines.append(f"  (tidak ada contoh di data test: {', '.join(missing)})")
    report = "\n".join(lines) + "\n"
    Path(f"hasil_praktik_{nim_safe}.txt").write_text(report)
    with open(f"hasil_praktik_{nim_safe}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nim", "nama", "waktu", "komputer", "sidik_jari_label", "eksperimen", "perubahan", "mAP50", "mAP50-95",
                    "P", "R", "ms_per_gambar", "menit", "model_terpilih", "split"])
        for n in EXPERIMENTS:
            r = rows[n]
            w.writerow([args.nim, args.nama, stamp, host, fp, n, r["change"], f"{r['mAP50']:.4f}", f"{r['mAP50-95']:.4f}",
                        f"{r['P']:.4f}", f"{r['R']:.4f}", f"{r['ms']:.2f}", f"{r['minutes']:.2f}", chosen == n, "val"])
        w.writerow([args.nim, args.nama, stamp, host, fp, chosen, rows[chosen]["change"], f"{test['mAP50']:.4f}",
                    f"{test['mAP50-95']:.4f}", f"{test['P']:.4f}", f"{test['R']:.4f}", f"{test['ms']:.2f}", "", True, "test"])
    print("\n" + report)
    print(f"Disimpan: hasil_praktik_{nim_safe}.txt dan .csv; folder {out}/")


if __name__ == "__main__":
    main()
