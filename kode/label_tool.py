#!/usr/bin/env python3
"""Alat label kotak sederhana (format YOLO), jalan di Pi (lewat VNC) atau PC. Hanya butuh OpenCV.

    python label_tool.py --dir dataset_raw --names mask glasses gloves

Mouse:  tarik dengan tombol KIRI = gambar kotak (kelas aktif)   |  klik KANAN = hapus kotak di titik itu
Tombol: 1 / 2 / 3 = pilih kelas (mask / glasses / gloves)       |  SPASI atau n = simpan & foto berikutnya
        p = foto sebelumnya   u = batalkan kotak terakhir   c = hapus semua kotak foto ini   q = simpan & keluar

        r = salin kotak dari foto sebelumnya (berguna kalau posisinya mirip, lalu geser/ganti seperlunya)
        x = lanjut walau kotak belum lengkap

Alat membaca nama file (mis. kacamata_hitam_masker_...) dan tahu APD apa yang HARUS ada. SPASI ditolak kalau
kotaknya belum ada, supaya tidak ada foto yang tak sengaja lolos tanpa label.
Foto TANPA APD (nama berisi "tanpa"): langsung tekan SPASI tanpa menggambar; tersimpan sebagai contoh negatif.
Foto yang sudah dilabeli dibuka lagi dengan kotaknya, jadi boleh dilanjutkan di lain waktu.
"""
import argparse
import os

IMG_EXT = (".jpg", ".jpeg", ".png")
COLORS = [(255, 160, 0), (255, 0, 200), (0, 200, 255), (0, 220, 0), (0, 0, 255)]   # BGR


def boxes_to_yolo(boxes, w, h):
    """boxes: [(cls, x1, y1, x2, y2)] dalam piksel -> teks label YOLO ternormalisasi."""
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
        y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        lines.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return "".join(line + "\n" for line in lines)


def yolo_to_boxes(text, w, h):
    boxes = []
    for line in text.splitlines():
        f = line.split()
        if len(f) != 5:
            continue
        c, xc, yc, bw, bh = int(f[0]), *(float(v) for v in f[1:])
        boxes.append((c, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
    return boxes


def expected_classes(filename, names):
    """APD yang seharusnya ada di foto, dari nama file: masker->mask, kacamata->glasses, sarung->gloves."""
    low = filename.lower()
    rules = {"mask": ("masker", "mask"), "glasses": ("kacamata", "glasses"), "gloves": ("sarung", "glove")}
    return {names.index(k) for k, words in rules.items() if k in names and any(w in low for w in words)}


def main():
    import cv2

    p = argparse.ArgumentParser(description="Alat label kotak YOLO")
    p.add_argument("--dir", required=True, help="folder foto (file .txt label disimpan di folder yang sama)")
    p.add_argument("--names", nargs="+", default=["mask", "glasses", "gloves"])
    p.add_argument("--max-size", type=int, default=900, help="sisi terpanjang tampilan (piksel)")
    p.add_argument("--only-new", action="store_true", help="lewati foto yang sudah punya file label")
    p.add_argument("--match", default="", help="hanya foto yang namanya mengandung teks ini, mis. --match kacamata_hitam_masker")
    p.add_argument("--todo", action="store_true",
                   help="hanya foto yang seharusnya punya APD (dari nama file) tapi label-nya belum ada atau kosong")
    args = p.parse_args()

    files = sorted(f for f in os.listdir(args.dir) if f.lower().endswith(IMG_EXT))
    if args.match:
        files = [f for f in files if args.match in f]
    if args.only_new:
        files = [f for f in files if not os.path.exists(os.path.join(args.dir, os.path.splitext(f)[0] + ".txt"))]
    if args.todo:
        def needs(f):
            t = os.path.join(args.dir, os.path.splitext(f)[0] + ".txt")
            empty = (not os.path.exists(t)) or os.path.getsize(t) == 0
            return bool(expected_classes(f, args.names)) and empty
        files = [f for f in files if needs(f)]
    if not files:
        raise SystemExit("Tidak ada foto untuk dilabeli.")

    state = {"cls": 0, "boxes": [], "drag": None, "mouse": (0, 0), "scale": 1.0, "prev": [], "msg": ""}

    def on_mouse(event, x, y, flags, _):
        s = state["scale"]
        state["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
            x0, y0 = state["drag"]
            state["drag"] = None
            if abs(x - x0) > 3 and abs(y - y0) > 3:
                state["boxes"].append((state["cls"], x0 / s, y0 / s, x / s, y / s))
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i in range(len(state["boxes"]) - 1, -1, -1):
                _, x1, y1, x2, y2 = state["boxes"][i]
                if min(x1, x2) * s <= x <= max(x1, x2) * s and min(y1, y2) * s <= y <= max(y1, y2) * s:
                    del state["boxes"][i]
                    break

    win = "Label APD"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    idx = 0
    while 0 <= idx < len(files):
        path = os.path.join(args.dir, files[idx])
        txt = os.path.splitext(path)[0] + ".txt"
        img = cv2.imread(path)
        if img is None:
            idx += 1
            continue
        h, w = img.shape[:2]
        state["scale"] = min(1.0, args.max_size / max(h, w))
        state["boxes"] = yolo_to_boxes(open(txt).read(), w, h) if os.path.exists(txt) else []
        state["msg"] = ""
        expected = expected_classes(files[idx], args.names)
        base = cv2.resize(img, (int(w * state["scale"]), int(h * state["scale"])))
        action = None
        while action is None:
            view = base.copy()
            s = state["scale"]
            for cls, x1, y1, x2, y2 in state["boxes"]:
                col = COLORS[cls % len(COLORS)]
                cv2.rectangle(view, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), col, 2)
                cv2.putText(view, args.names[cls], (int(min(x1, x2) * s), int(min(y1, y2) * s) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            if state["drag"]:
                cv2.rectangle(view, state["drag"], state["mouse"], COLORS[state["cls"] % len(COLORS)], 1)
            bar = (f"{idx + 1}/{len(files)}  {files[idx]}  | kelas aktif: {state['cls'] + 1}={args.names[state['cls']]}"
                   f"  | kotak: {len(state['boxes'])}  | SPASI=simpan+lanjut  p=mundur  u=undo  c=hapus  q=keluar")
            have = {b[0] for b in state["boxes"]}
            need_txt = ", ".join(args.names[c] + ("  OK" if c in have else "  BELUM") for c in sorted(expected)) or "tidak ada (foto tanpa APD)"
            cv2.rectangle(view, (0, 0), (view.shape[1], 44), (30, 30, 30), -1)
            cv2.putText(view, bar, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(view, "Harus ada: " + need_txt + "   r=salin dari foto sebelumnya   x=lanjut paksa", (6, 37),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if state["msg"]:
                cv2.rectangle(view, (0, 44), (view.shape[1], 70), (0, 0, 160), -1)
                cv2.putText(view, state["msg"], (6, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(win, view)
            k = cv2.waitKey(20) & 0xFF
            if k != 255:
                state["msg"] = ""
            if k in (ord(" "), ord("n"), ord("x")):
                missing = [args.names[c] for c in sorted(expected) if c not in {b[0] for b in state["boxes"]}]
                if missing and k != ord("x"):
                    state["msg"] = "BELUM ADA kotak: " + ", ".join(missing) + "  (gambar dulu, atau tekan x untuk lanjut tanpa itu)"
                else:
                    action = "next"
            elif k == ord("r") and state["prev"]:
                state["boxes"] = list(state["prev"])
            elif k == ord("p"):
                action = "prev"
            elif k == ord("q"):
                action = "quit"
            elif k == ord("u") and state["boxes"]:
                state["boxes"].pop()
            elif k == ord("c"):
                state["boxes"] = []
            elif ord("1") <= k < ord("1") + len(args.names):
                state["cls"] = k - ord("1")
        open(txt, "w").write(boxes_to_yolo(state["boxes"], w, h))
        if state["boxes"]:
            state["prev"] = list(state["boxes"])
        if action == "quit":
            break
        idx += 1 if action == "next" else -1 if idx > 0 else 0
    cv2.destroyAllWindows()
    done = sum(os.path.exists(os.path.join(args.dir, os.path.splitext(f)[0] + ".txt")) for f in os.listdir(args.dir) if f.lower().endswith(IMG_EXT))
    filled = sum(1 for f in os.listdir(args.dir) if f.endswith(".txt") and os.path.getsize(os.path.join(args.dir, f)) > 0)
    print(f"Selesai. {done} foto punya file label di '{args.dir}', {filled} di antaranya berisi kotak "
          f"(sisanya kosong = foto tanpa APD).")


if __name__ == "__main__":
    main()
