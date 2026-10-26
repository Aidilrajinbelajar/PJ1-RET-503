#!/usr/bin/env python3
"""Ambil foto dataset dari webcam / kamera Pi.

SPASI = simpan foto, q = keluar.  --auto 2 = simpan otomatis tiap 2 detik.

Contoh:
    python collect_data.py --source 0 --out dataset_raw --tag lengkap_terang
    python collect_data.py --source 0 --out dataset_raw --tag tanpa_apd_redup --auto 2

Foto disimpan sebagai yang kamu lihat di layar (setelah --flip). Pakai --tag untuk mencatat kondisinya
(misalnya masker_terang, kacamata_redup, tanpa_apd) supaya nanti mudah dicek variasinya.
"""
import argparse
import os
import time

import cv2

FLIPS = {"none": None, "h": 1, "v": 0, "both": -1}


def apply_flip(frame, mode):
    code = FLIPS.get(mode)
    return frame if code is None else cv2.flip(frame, code)


class Cam:
    """Kamera sederhana: webcam (angka), file video/gambar, atau 'picam' (hanya di Raspberry Pi). Tanpa ultralytics."""

    def __init__(self, src, width, height):
        self.pi = None
        self.cap = None
        if src == "picam":
            from picamera2 import Picamera2
            self.pi = Picamera2()
            self.pi.configure(self.pi.create_preview_configuration(main={"size": (width, height), "format": "RGB888"}))
            self.pi.start()
            return
        idx = int(src) if src.isdigit() else src
        backend = cv2.CAP_DSHOW if (os.name == "nt" and isinstance(idx, int)) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(idx, backend)
        if not self.cap.isOpened():
            raise SystemExit(f"Tidak bisa membuka kamera '{src}'. Coba angka lain (--source 1) atau cek kabel/izin kamera.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        if self.pi is not None:
            return True, self.pi.capture_array()
        return self.cap.read()

    def close(self):
        if self.pi is not None:
            self.pi.stop()
        elif self.cap is not None:
            self.cap.release()


def count_images(folder):
    return sum(1 for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def main():
    p = argparse.ArgumentParser(description="Pengambil foto dataset APD")
    p.add_argument("--source", default="0", help="angka webcam, 'picam', path video, atau path gambar")
    p.add_argument("--out", default="dataset_raw", help="folder tujuan foto")
    p.add_argument("--tag", default="foto", help="penanda kondisi, ikut masuk ke nama file")
    p.add_argument("--auto", type=float, default=0, help="simpan otomatis tiap N detik (0 = manual dengan SPASI)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--flip", choices=["none", "h", "v", "both"], default="none")
    p.add_argument("--quality", type=int, default=95, help="kualitas JPEG 1-100")
    p.add_argument("--max-frames", type=int, default=0, help="berhenti setelah N frame (0 = tanpa batas)")
    p.add_argument("--headless", action="store_true", help="tanpa jendela; wajib dengan --auto")
    args = p.parse_args()

    if args.headless and args.auto <= 0:
        raise SystemExit("--headless harus dipakai bersama --auto (tidak ada tombol untuk menyimpan).")
    os.makedirs(args.out, exist_ok=True)
    safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.tag) or "foto"

    src = Cam(args.source, args.width, args.height)
    saved, n_frames, last_save = 0, 0, 0.0

    def save(img):
        nonlocal saved
        name = f"{safe_tag}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{saved:04d}.jpg"
        cv2.imwrite(os.path.join(args.out, name), img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        saved += 1

    try:
        while True:
            ok, frame = src.read()
            if not ok:
                break
            frame = apply_flip(frame, args.flip)
            n_frames += 1

            if args.auto > 0 and time.time() - last_save >= args.auto:
                save(frame)
                last_save = time.time()

            key = -1
            if not args.headless:
                view = frame.copy()
                info = f"tag: {safe_tag} | sesi ini: {saved} | total di folder: {count_images(args.out)}"
                cv2.rectangle(view, (0, 0), (view.shape[1], 34), (30, 30, 30), -1)
                cv2.putText(view, info + "   [SPASI] simpan  [q] keluar", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1)
                cv2.imshow("Ambil dataset", view)
                key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                save(frame)
            elif key == ord("q"):
                break
            if args.max_frames and n_frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        cv2.destroyAllWindows()
    print(f"Selesai. {saved} foto baru disimpan di '{args.out}' (total {count_images(args.out)}).")


if __name__ == "__main__":
    main()
