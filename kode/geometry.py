"""Geometri untuk menentukan area wajah dan tangan dari keypoint pose (format COCO 17 titik).

Hanya butuh numpy, jadi bisa diuji tanpa model.
"""
from __future__ import annotations

import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = range(11)


def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    return (max(0, int(x1)), max(0, int(y1)), min(int(w), int(x2)), min(int(h), int(y2)))


def square_roi(cx, cy, side, w, h):
    half = side / 2.0
    return clip_box((cx - half, cy - half, cx + half, cy + half), w, h)


def face_roi(kps, conf, w, h, kp_thr=0.5, min_side=48):
    """Area wajah untuk dicek masker/kacamata.

    Mengembalikan (roi, alasan). roi=None kalau wajah tidak bisa dinilai:
      "not_visible" -> mata tidak terlihat (membelakangi / tertutup)
      "too_small"   -> wajah terlalu kecil/jauh untuk dinilai
    """
    if conf[L_EYE] < kp_thr or conf[R_EYE] < kp_thr:
        return None, "not_visible"
    eye_dist = _dist(kps[L_EYE], kps[R_EYE])
    if eye_dist < 1e-3:
        return None, "not_visible"
    ex = (kps[L_EYE][0] + kps[R_EYE][0]) / 2.0
    ey = (kps[L_EYE][1] + kps[R_EYE][1]) / 2.0
    side = 4.2 * eye_dist          # kepala dari dahi sampai dagu + sedikit margin
    if side < min_side:
        return None, "too_small"
    roi = square_roi(ex, ey + 0.25 * eye_dist, side, w, h)
    if roi[2] - roi[0] < min_side * 0.6 or roi[3] - roi[1] < min_side * 0.6:
        return None, "not_visible"  # wajah terpotong di tepi frame
    return roi, "ok"


def hand_rois(kps, conf, w, h, kp_thr=0.4, min_side=40):
    """Area tangan untuk dicek sarung tangan. Mengembalikan list (nama, roi, alasan).

    Tangan yang pergelangannya tidak terlihat tidak dimasukkan sama sekali (artinya "tidak terlihat").
    Ukuran area diturunkan dari panjang lengan bawah (siku-pergelangan), fallback lebar bahu.
    """
    out = []
    shoulders_ok = conf[L_SHO] >= kp_thr and conf[R_SHO] >= kp_thr
    sho_w = _dist(kps[L_SHO], kps[R_SHO]) if shoulders_ok else 0.0
    for name, wri, elb in (("kiri", L_WRI, L_ELB), ("kanan", R_WRI, R_ELB)):
        if conf[wri] < kp_thr:
            continue
        wrist = np.asarray(kps[wri], dtype=float)
        if conf[elb] >= kp_thr:
            forearm = _dist(kps[elb], wrist)
            direction = (wrist - np.asarray(kps[elb], dtype=float)) / max(forearm, 1e-6)
        elif sho_w > 0:
            forearm = 0.8 * sho_w
            direction = np.zeros(2)
        else:
            continue
        side = 1.0 * forearm
        if side < min_side:
            out.append((name, None, "too_small"))
            continue
        center = wrist + direction * 0.3 * forearm   # telapak ada sedikit di luar pergelangan
        roi = square_roi(center[0], center[1], side, w, h)
        if roi[2] - roi[0] < min_side * 0.6 or roi[3] - roi[1] < min_side * 0.6:
            continue
        out.append((name, roi, "ok"))
    return out


def face_anchor(kps, conf, kp_thr=0.5):
    """(ex, ey, eye_dist): titik tengah kedua mata dan jarak antar mata. None kalau mata tidak terlihat."""
    if conf[L_EYE] < kp_thr or conf[R_EYE] < kp_thr:
        return None
    d = _dist(kps[L_EYE], kps[R_EYE])
    if d < 1e-3:
        return None
    return ((kps[L_EYE][0] + kps[R_EYE][0]) / 2.0, (kps[L_EYE][1] + kps[R_EYE][1]) / 2.0, d)


def eye_roi(anchor, w, h):
    """Potongan persegi panjang hanya di sekitar mata (masker di bawahnya tidak ikut). None kalau terlalu kecil."""
    ex, ey, d = anchor
    roi = clip_box((ex - 1.7 * d, ey - 0.8 * d, ex + 1.7 * d, ey + 0.9 * d), w, h)
    if roi[2] - roi[0] < 32 or roi[3] - roi[1] < 16:
        return None
    return roi


def plausible_on_face(key, box, anchor):
    """Apakah kotak kacamata/masker masuk akal terhadap posisi mata? Mengembalikan (ok, alasan).

    Kacamata: pusatnya dekat garis mata, lebar kira-kira 2-4 kali jarak antar mata.
    Masker: pusatnya di bawah garis mata (area hidung sampai dagu).
    Ini menyaring deteksi liar (rambut, bayangan, latar belakang) yang membuat status berkedip.
    """
    ex, ey, d = anchor
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if key == "glasses":
        if abs(cx - ex) > 0.9 * d or abs(cy - ey) > 0.7 * d:
            return False, "posisi"
        if not (1.6 * d <= w <= 4.5 * d) or h > 2.5 * d:
            return False, "ukuran"
    elif key == "mask":
        if abs(cx - ex) > 1.2 * d or not (0.6 * d <= cy - ey <= 2.4 * d):
            return False, "posisi"
        if not (1.4 * d <= w <= 4.5 * d) or not (0.6 * d <= h <= 3.0 * d):
            return False, "ukuran"
    return True, "ok"


def plausible_in_roi(box, roi):
    """Sarung tangan harus berpusat di area tangan dan ukurannya wajar terhadap area itu."""
    rx1, ry1, rx2, ry2 = roi
    side = max(rx2 - rx1, ry2 - ry1)
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
        return False, "posisi"
    if (x2 - x1) > 1.15 * side or (y2 - y1) > 1.15 * side:
        return False, "ukuran"
    if (x2 - x1) * (y2 - y1) < 0.04 * (rx2 - rx1) * (ry2 - ry1):
        return False, "ukuran"
    return True, "ok"
