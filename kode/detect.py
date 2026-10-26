#!/usr/bin/env python3
"""Deteksi APD (masker, kacamata, sarung tangan) real-time di Raspberry Pi 5.

Contoh:
    python detect.py --model best_ncnn_model --source picam --imgsz 416
    python detect.py --model best.pt --source 0            # webcam USB
    python detect.py --model best_ncnn_model --source picam --headless
"""
import argparse
import time

import cv2
from ultralytics import YOLO

from ppe_logic import ITEMS, Det, Smoother, build_class_map, observe, overall

LABEL = {"mask": "MASKER", "glasses": "KACAMATA", "gloves": "SARUNG TANGAN"}
STATE_TEXT = {"OK": "SUDAH", "MISSING": "BELUM", "CHECKING": "MEMERIKSA", "UNSEEN": "TIDAK TERLIHAT"}
STATE_COLOR = {"OK": (0, 200, 0), "MISSING": (0, 0, 255), "CHECKING": (0, 200, 255), "UNSEEN": (170, 170, 170)}
OVERALL_TEXT = {"OK": "LENGKAP - BOLEH MASUK", "VIOLATION": "APD BELUM LENGKAP", "CHECKING": "MEMERIKSA...",
                "UNSEEN": "HADAP KAMERA, TUNJUKKAN TANGAN"}
OVERALL_COLOR = {"OK": (0, 200, 0), "VIOLATION": (0, 0, 255), "CHECKING": (0, 200, 255), "UNSEEN": (170, 170, 170)}
ITEM_COLOR = {"mask": (255, 160, 0), "glasses": (255, 0, 200), "gloves": (0, 220, 220)}


def on_result(pid, state, statuses):
    """Dipanggil HANYA saat status keseluruhan berubah.

    Isi dengan aksi sungguhan, misalnya:
      - nyalakan buzzer / LED lewat GPIO (gpiozero)
      - publish ke MQTT supaya ESP32 buka/tutup pintu
    """
    detail = ", ".join(f"{LABEL[k]}={STATE_TEXT[v]}" for k, v in statuses.items())
    print(f"[{time.strftime('%H:%M:%S')}] orang #{pid}: {state} ({detail})", flush=True)


class Source:
    def __init__(self, src, width, height):
        self.picam = None
        self.cap = None
        self.image = None
        if src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            self.image = cv2.imread(src)
            if self.image is None:
                raise SystemExit(f"Gambar tidak bisa dibuka: {src}")
        elif src == "picam":
            from picamera2 import Picamera2

            self.picam = Picamera2()
            cfg = self.picam.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
            self.picam.configure(cfg)
            self.picam.start()
        else:
            self.cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self.cap.isOpened():
                raise SystemExit(f"Tidak bisa membuka sumber video: {src}")

    def read(self):
        if self.image is not None:
            return True, self.image.copy()
        if self.picam is not None:
            return True, self.picam.capture_array()  # format RGB888 di Picamera2 = urutan BGR, cocok untuk OpenCV
        return self.cap.read()

    def close(self):
        if self.picam is not None:
            self.picam.stop()
        if self.cap is not None:
            self.cap.release()


def draw_box(frame, box, color, text):
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_panel(frame, x, y, pid, statuses, state, required, title=None):
    lines = [title or f"ORANG #{pid}"] + [f"{LABEL[k]}: {STATE_TEXT[statuses[k]]}" for k in required]
    h = 26 * (len(lines) + 1) + 8
    cv2.rectangle(frame, (x, y), (x + 290, y + h), (30, 30, 30), -1)
    cv2.putText(frame, lines[0], (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for i, k in enumerate(required, start=1):
        cv2.putText(frame, lines[i], (x + 8, y + 22 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    STATE_COLOR[statuses[k]], 2)
    cv2.putText(frame, OVERALL_TEXT[state], (x + 8, y + 22 + 26 * (len(lines))), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                OVERALL_COLOR[state], 2)


FLIP_CODE = {"h": 1, "v": 0, "both": -1}


def apply_flip(frame, mode):
    """mode: none / h (cermin kiri-kanan) / v (terbalik atas-bawah) / both (putar 180 derajat)."""
    return frame if mode == "none" else cv2.flip(frame, FLIP_CODE[mode])


def parse_args():
    p = argparse.ArgumentParser(description="Deteksi APD K3: masker, kacamata, sarung tangan")
    p.add_argument("--model", required=True, help="best.pt atau folder hasil ekspor NCNN (best_ncnn_model)")
    p.add_argument("--source", default="picam", help="'picam' untuk kamera Pi, angka (0,1) untuk webcam, atau path video")
    p.add_argument("--imgsz", type=int, default=416, help="ukuran input; 320-480 cocok untuk Pi 5")
    p.add_argument("--conf", type=float, default=0.25, help="ambang dasar model; ambang per-kelas ada di ppe_logic.py")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--require", nargs="+", choices=ITEMS, default=list(ITEMS), help="APD yang wajib dipakai")
    p.add_argument("--gloves-count", type=int, default=1, help="jumlah sarung tangan minimal yang harus terlihat (1 atau 2)")
    p.add_argument("--window", type=int, default=8, help="jumlah frame untuk smoothing")
    p.add_argument("--need", type=int, default=5, help="minimal frame (dari window) yang harus mendeteksi")
    p.add_argument("--flip", choices=["none", "h", "v", "both"], default="none",
                   help="h = balik kiri-kanan (cermin), v = balik atas-bawah, both = putar 180 derajat")
    p.add_argument("--headless", action="store_true", help="tanpa jendela tampilan (untuk SSH atau service)")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model, task="detect")
    class_map = build_class_map(model.names)
    found = set(class_map.values())
    missing_cls = [k for k in args.require if k not in found]
    if missing_cls:
        raise SystemExit(
            f"Model tidak punya kelas untuk: {missing_cls}\nKelas di model: {dict(model.names)}\n"
            "Latih ulang dengan dataset yang berisi kelas tersebut, atau kurangi --require."
        )
    has_person = "person" in found
    mode = "multi-orang (ada kelas person)" if has_person else "pos pemeriksaan (satu orang di depan kamera)"
    print(f"Mode: {mode} | APD wajib: {args.require} | kelas terpakai: {class_map}", flush=True)

    smoother = Smoother(window=args.window, need=args.need, required_gloves=args.gloves_count)
    last_state = {}
    src = Source(args.source, args.width, args.height)
    fps, t_prev = 0.0, time.time()

    try:
        while True:
            ok, frame = src.read()
            if not ok:
                print("Frame kosong, berhenti.")
                break
            frame = apply_flip(frame, args.flip)

            if has_person:
                res = model.track(frame, imgsz=args.imgsz, conf=args.conf, persist=True,
                                  tracker="bytetrack.yaml", verbose=False)[0]
            else:
                res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

            dets, persons = [], []
            for i, b in enumerate(res.boxes):
                key = class_map.get(int(b.cls[0]))
                if key is None:
                    continue
                box = tuple(float(v) for v in b.xyxy[0].tolist())
                if key == "person":
                    pid = int(b.id[0]) if b.id is not None else i
                    persons.append((pid, box))
                else:
                    dets.append(Det(key, float(b.conf[0]), box))
                    draw_box(frame, box, ITEM_COLOR[key], f"{LABEL[key]} {float(b.conf[0]):.2f}")

            smoother.tick()
            targets = persons if has_person else [(0, None)]
            for n, (pid, pbox) in enumerate(targets):
                statuses = smoother.update(pid, observe(dets, person_box=pbox))
                state = overall(statuses, args.require)
                if last_state.get(pid) != state:
                    last_state[pid] = state
                    on_result(pid, state, {k: statuses[k] for k in args.require})
                if pbox is not None:
                    cv2.rectangle(frame, tuple(int(v) for v in pbox[:2]), tuple(int(v) for v in pbox[2:]),
                                  OVERALL_COLOR[state], 2)
                    draw_panel(frame, 10 + 300 * n, 10, pid, statuses, state, args.require)
                else:
                    draw_panel(frame, 10, 10, pid, statuses, state, args.require, title="PEMERIKSAAN K3")

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            cv2.putText(frame, f"FPS {fps:.1f}", (frame.shape[1] - 120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            if not args.headless:
                cv2.imshow("K3 APD - tekan q untuk keluar", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
