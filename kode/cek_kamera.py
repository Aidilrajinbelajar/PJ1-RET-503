#!/usr/bin/env python3
"""Cek nomor kamera yang tersedia. Menyimpan satu foto per kamera (kamera_0.jpg, kamera_1.jpg, ...).

    python cek_kamera.py

Buka foto-fotonya: yang tajam, sudut pandang sama dengan di Raspberry Pi, itulah webcam Logitech C270.
Nomor di nama file = nilai untuk --source di collect_data.py.
"""
import os
import sys

import cv2


def main():
    max_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", getattr(cv2, "CAP_MSMF", cv2.CAP_ANY))] if os.name == "nt" \
        else [("DEFAULT", cv2.CAP_ANY)]
    found = 0
    for name, be in backends:
        print(f"--- backend {name} ---")
        for i in range(max_idx):
            cap = cv2.VideoCapture(i, be)
            if not cap.isOpened():
                print(f"  nomor {i}: tidak ada")
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok = False
            frame = None
            for _ in range(8):                 # beberapa frame pertama sering gelap
                ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                fn = f"kamera_{name}_{i}.jpg"
                cv2.imwrite(fn, frame)
                print(f"  nomor {i}: BISA dibuka, ukuran {w}x{h}, foto disimpan: {fn}")
                found += 1
            else:
                print(f"  nomor {i}: terbuka tapi tidak ada gambar (dipakai aplikasi lain?)")
            cap.release()
    print(f"\nSelesai. {found} kamera menghasilkan gambar. Buka file kamera_*.jpg untuk melihat mana yang webcam C270.")


if __name__ == "__main__":
    main()
