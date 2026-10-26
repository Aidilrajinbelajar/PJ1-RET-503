#!/usr/bin/env python3
"""Ekspor best.pt ke NCNN (format tercepat di CPU Raspberry Pi). Jalankan DI Raspberry Pi.

    python export_ncnn.py best.pt --imgsz 416
Hasil: folder best_ncnn_model/ -> dipakai di detect.py --model best_ncnn_model
"""
import argparse

from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("weights")
p.add_argument("--imgsz", type=int, default=416, help="harus sama dengan --imgsz saat detect.py")
a = p.parse_args()

out = YOLO(a.weights).export(format="ncnn", imgsz=a.imgsz)
print("Selesai:", out)
