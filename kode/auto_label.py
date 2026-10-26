#!/usr/bin/env python3
"""Pra-label otomatis: model menebak kotak APD di foto, hasilnya ditulis sebagai file label YOLO (.txt).
Setelah itu KAMU memeriksanya dengan label_tool.py (tekan SPASI di foto yang benar, perbaiki yang salah).

Jalankan di Raspberry Pi (ultralytics dan model sudah ada di sana):
    python auto_label.py --dir dataset_raw --pose-model yolo11n-pose.pt --ppe-model ppe_best.pt

Aturan (APD yang diharapkan dibaca dari nama file, sama seperti label_tool.py):
  * Kotak dari model dipakai kalau keyakinannya cukup.
  * Kacamata yang diharapkan tapi tidak terdeteksi: kotak TEBAKAN dari posisi mata (ditandai "tebakan" di laporan).
  * Masker/sarung tangan yang diharapkan tapi tidak terdeteksi: dibiarkan kosong untuk digambar manual.
  * Foto "tanpa APD": label kosong (contoh negatif).
Foto yang sudah punya kotak tidak ditimpa (kecuali --overwrite). Laporan: auto_label_report.csv
"""
import argparse
import csv
import os

from label_tool import expected_classes

IMG_EXT = (".jpg", ".jpeg", ".png")


def guess_glasses_box(anchor, w, h):
    """Kotak kacamata perkiraan dari posisi mata: lebar sekitar 2x jarak mata, tinggi sekitar 0,7x."""
    ex, ey, d = anchor
    x1, x2 = max(0, ex - 1.0 * d), min(w, ex + 1.0 * d)
    y1, y2 = max(0, ey - 0.35 * d), min(h, ey + 0.35 * d)
    return (x1, y1, x2, y2)


def yolo_line(cls, box, w, h):
    x1, y1, x2, y2 = box
    return f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n"


def main():
    import cv2
    from ultralytics import YOLO

    from ppe_logic import ITEMS, build_class_map
    from smart_vision import PoseGuided

    p = argparse.ArgumentParser(description="Pra-label otomatis foto APD")
    p.add_argument("--dir", required=True)
    p.add_argument("--pose-model", default="yolo11n-pose.pt")
    p.add_argument("--ppe-model", required=True)
    p.add_argument("--conf", type=float, default=0.10, help="keyakinan minimum model (rendah, supaya kandidat tidak terbuang)")
    p.add_argument("--min-mask", type=float, default=0.40)
    p.add_argument("--min-glasses", type=float, default=0.15)
    p.add_argument("--min-gloves", type=float, default=0.30)
    p.add_argument("--no-guess", action="store_true", help="jangan membuat kotak tebakan untuk kacamata")
    p.add_argument("--overwrite", action="store_true", help="timpa juga file label yang sudah berisi kotak")
    args = p.parse_args()

    ppe = YOLO(args.ppe_model)
    class_map = build_class_map(ppe.names)
    engine = PoseGuided(YOLO(args.pose_model), ppe, class_map, pose_imgsz=416, crop_imgsz=320, conf=args.conf,
                        max_people=3, geometry_filter=True, eye_crop=True,
                        min_conf={"mask": args.min_mask, "glasses": args.min_glasses, "gloves": args.min_gloves})
    names = list(ITEMS)

    files = sorted(f for f in os.listdir(args.dir) if f.lower().endswith(IMG_EXT))
    rows, stats = [], {"model": 0, "tebakan": 0, "manual": 0, "negatif": 0, "dilewati": 0}
    for f in files:
        path = os.path.join(args.dir, f)
        txt = os.path.splitext(path)[0] + ".txt"
        if os.path.exists(txt) and os.path.getsize(txt) > 0 and not args.overwrite:
            stats["dilewati"] += 1
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        expected = expected_classes(f, names)
        lines, src = [], {}
        if not expected:
            open(txt, "w").write("")
            stats["negatif"] += 1
            rows.append([f, "negatif", "", ""])
            continue
        people = engine.analyze(img)
        with_face = [q for q in people if q.anchor is not None]        # orang terbesar yang matanya terlihat
        pr = with_face[0] if with_face else (people[0] if people else None)
        for cls in sorted(expected):
            key = names[cls]
            dets = [d for d in (pr.dets if pr else []) if d.key == key and d.conf >= engine.thr[key]]
            dets.sort(key=lambda d: -d.conf)
            if key == "gloves":
                dets = dets[:2]
            else:
                dets = dets[:1]
            if dets:
                for d in dets:
                    lines.append(yolo_line(cls, d.box, w, h))
                src[key] = f"model {dets[0].conf:.2f}"
            elif key == "glasses" and pr is not None and pr.anchor is not None and not args.no_guess:
                lines.append(yolo_line(cls, guess_glasses_box(pr.anchor, w, h), w, h))
                src[key] = "tebakan"
            else:
                src[key] = "manual"
        open(txt, "w").write("".join(lines))
        for k, v in src.items():
            stats["model" if v.startswith("model") else v] += 1
        rows.append([f, "; ".join(f"{k}={v}" for k, v in src.items()), len(lines), ""])
    with open(os.path.join(args.dir, "auto_label_report.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["file", "sumber_kotak", "jumlah_kotak", ""])
        wr.writerows(rows)
    print(f"Selesai. Kotak dari model: {stats['model']} | tebakan posisi mata: {stats['tebakan']} | "
          f"perlu digambar manual: {stats['manual']} | foto tanpa APD (kosong): {stats['negatif']} | "
          f"dilewati (sudah berlabel): {stats['dilewati']}")
    print("Berikutnya periksa semuanya: python label_tool.py --dir", args.dir)


if __name__ == "__main__":
    main()
