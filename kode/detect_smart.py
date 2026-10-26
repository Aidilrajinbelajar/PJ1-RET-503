#!/usr/bin/env python3
"""Deteksi APD versi pintar: pose -> area wajah & tangan -> deteksi APD di potongan yang diperbesar.

Contoh di Pi 5 (model sudah diekspor ke NCNN):
    python detect_smart.py --pose-model yolo11n-pose_ncnn_model --ppe-model best_ncnn_model --source picam

Uji dengan gambar:
    python detect_smart.py --pose-model yolo11n-pose.pt --ppe-model best.pt --source foto.jpg \
        --max-frames 8 --headless --save-last hasil.jpg
"""
import argparse
import os
import time

import cv2
from ultralytics import YOLO

from detect import (ITEM_COLOR, LABEL, OVERALL_COLOR, STATE_COLOR, STATE_TEXT, Source, apply_flip, draw_box,
                    draw_panel, on_result)
from ppe_logic import ITEMS, Smoother, build_class_map, overall
from smart_vision import PoseGuided


def save_crops(raw, pr, folder):
    """Simpan potongan wajah & tangan apa adanya (tanpa gambar kotak) untuk dilabeli lalu dipakai training."""
    os.makedirs(folder, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    areas = [("wajah", pr.face_roi)] + [(f"tangan_{name}", roi) for name, roi, _ in pr.hands]
    for label, roi in areas:
        if roi is None:
            continue
        x1, y1, x2, y2 = roi
        cv2.imwrite(os.path.join(folder, f"{stamp}_p{pr.pid}_{label}.jpg"), raw[y1:y2, x1:x2])


def parse_min_conf(items):
    """['mask=0.6', 'glasses=0.55'] -> {'mask': 0.6, 'glasses': 0.55}"""
    out = {}
    for it in items or []:
        key, _, val = it.partition("=")
        if key not in ITEMS or not val:
            raise SystemExit(f"--min-conf salah: '{it}'. Contoh: --min-conf mask=0.6 glasses=0.6 gloves=0.4")
        out[key] = float(val)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Deteksi APD K3 (pose + crop)")
    p.add_argument("--pose-model", required=True, help="yolo11n-pose.pt atau folder NCNN-nya")
    p.add_argument("--ppe-model", required=True, help="model APD (best.pt atau folder NCNN)")
    p.add_argument("--source", default="picam", help="'picam', angka webcam, path video, atau path gambar")
    p.add_argument("--pose-imgsz", type=int, default=416)
    p.add_argument("--crop-imgsz", type=int, default=320, help="ukuran input model APD untuk potongan wajah/tangan")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--require", nargs="+", choices=ITEMS, default=list(ITEMS))
    p.add_argument("--max-people", type=int, default=2, help="jumlah orang terbesar yang dianalisis per frame")
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--need", type=int, default=5)
    p.add_argument("--flip", choices=["none", "h", "v", "both"], default="none",
                   help="h = balik kiri-kanan (cermin), v = balik atas-bawah, both = putar 180 derajat")
    p.add_argument("--min-conf", nargs="+", metavar="KELAS=NILAI",
                   help="ambang keyakinan per APD, contoh: --min-conf mask=0.6 glasses=0.6 gloves=0.4")
    p.add_argument("--no-geometry-filter", action="store_true",
                   help="matikan filter posisi/ukuran (kacamata harus di mata, masker di bawah mata, dst.)")
    p.add_argument("--eye-crop", action="store_true",
                   help="cek kacamata juga di potongan area mata saja (membantu kalau kacamata tak terbaca saat bermasker)")
    p.add_argument("--debug", action="store_true", help="tampilkan juga deteksi yang DITOLAK (abu-abu) dan cetak ke terminal")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-frames", type=int, default=0, help="berhenti setelah N frame (0 = tanpa batas)")
    p.add_argument("--save-last", default="", help="simpan frame terakhir yang sudah dianotasi ke file ini")
    p.add_argument("--save-crops", default="", help="folder untuk menyimpan potongan wajah/tangan (bahan labeling & training ulang)")
    p.add_argument("--save-every", type=int, default=15, help="simpan potongan tiap N frame")
    return p.parse_args()


def main():
    args = parse_args()
    pose = YOLO(args.pose_model, task="pose")
    ppe = YOLO(args.ppe_model, task="detect")
    class_map = build_class_map(ppe.names)
    missing_cls = [k for k in args.require if k not in set(class_map.values())]
    if missing_cls:
        raise SystemExit(f"Model APD tidak punya kelas untuk: {missing_cls}\nKelas di model: {dict(ppe.names)}")
    print(f"APD wajib: {args.require} | kelas terpakai: {class_map}", flush=True)

    engine = PoseGuided(pose, ppe, class_map, pose_imgsz=args.pose_imgsz, crop_imgsz=args.crop_imgsz,
                        conf=args.conf, max_people=args.max_people, min_conf=parse_min_conf(args.min_conf),
                        geometry_filter=not args.no_geometry_filter, eye_crop=args.eye_crop)
    smoother = Smoother(window=args.window, need=args.need)
    last_state = {}
    src = Source(args.source, args.width, args.height)
    fps, t_prev, n_frames, frame = 0.0, time.time(), 0, None

    try:
        while True:
            ok, frame = src.read()
            if not ok:
                break
            frame = apply_flip(frame, args.flip)
            n_frames += 1
            smoother.tick()

            raw = frame.copy() if args.save_crops else None
            for n, pr in enumerate(engine.analyze(frame)):
                statuses = smoother.update(pr.pid, pr.obs)
                state = overall(statuses, args.require)
                if last_state.get(pr.pid) != state:
                    last_state[pr.pid] = state
                    on_result(pr.pid, state, {k: statuses[k] for k in args.require})

                if raw is not None and n_frames % args.save_every == 0:
                    save_crops(raw, pr, args.save_crops)

                # kotak orang + area yang dinilai
                x1, y1, x2, y2 = (int(v) for v in pr.box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), OVERALL_COLOR[state], 2)
                if pr.face_roi is not None:
                    fc = STATE_COLOR["OK" if (pr.obs.mask and pr.obs.glasses) else "MISSING"]
                    draw_box(frame, pr.face_roi, fc, "wajah")
                for name, hroi, gloved in pr.hands:
                    draw_box(frame, hroi, STATE_COLOR["OK" if gloved else "MISSING"], f"tangan {name}")
                for d in pr.dets:
                    draw_box(frame, d.box, ITEM_COLOR[d.key], f"{LABEL[d.key]} {d.conf:.2f}")
                if args.debug:
                    for d, why in pr.rejected:
                        draw_box(frame, d.box, (120, 120, 120), f"x {LABEL[d.key]} {d.conf:.2f} ({why})")
                    if n_frames % 10 == 0:
                        acc = ", ".join(f"{d.key} {d.conf:.2f}" for d in pr.dets) or "-"
                        rej = ", ".join(f"{d.key} {d.conf:.2f} ({why})" for d, why in pr.rejected) or "-"
                        print(f"  dbg #{pr.pid}  diterima: {acc} | ditolak: {rej}", flush=True)
                draw_panel(frame, 10 + 300 * n, 10, pr.pid, statuses, state, args.require)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            cv2.putText(frame, f"FPS {fps:.1f}", (frame.shape[1] - 120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            if not args.headless:
                cv2.imshow("K3 APD pintar - tekan q untuk keluar", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and n_frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if args.save_last and frame is not None:
            cv2.imwrite(args.save_last, frame)
            print("Disimpan:", args.save_last)
        src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
