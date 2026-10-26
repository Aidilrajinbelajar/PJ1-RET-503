#!/usr/bin/env python3
"""Fine-tune YOLO untuk APD. Jalankan di PC/Colab yang punya GPU, BUKAN di Pi.

    python train.py --data /path/ke/data.yaml
    python train.py --data data.yaml --model yolo11n.pt --epochs 100 --imgsz 640
"""
import argparse

from ultralytics import YOLO

from ppe_logic import build_class_map


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path data.yaml (format YOLO)")
    p.add_argument("--model", default="yolo11n.pt", help="model dasar; n = paling ringan untuk Pi")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640, help="sarung tangan & kacamata kecil, latih di 640")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="ppe_k3")
    args = p.parse_args()

    model = YOLO(args.model)
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                          name=args.name, patience=25)

    # Cek apakah kelas hasil latihan terbaca oleh detect.py
    best = YOLO(f"{results.save_dir}/weights/best.pt")
    mapped = build_class_map(best.names)
    print("\nKelas di model :", dict(best.names))
    print("Terpetakan     :", mapped)
    wanted = {"mask", "glasses", "gloves"} - set(mapped.values())
    if wanted:
        print(f"PERINGATAN: kelas {sorted(wanted)} tidak dikenali. Ganti nama kelas di data.yaml "
              "atau tambahkan alias di ppe_logic.py")
    print(f"\nModel terbaik: {results.save_dir}/weights/best.pt  -> salin ke Raspberry Pi")


if __name__ == "__main__":
    main()
