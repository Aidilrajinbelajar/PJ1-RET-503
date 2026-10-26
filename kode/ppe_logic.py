"""Logika cek APD (masker, kacamata, sarung tangan).

Modul ini tidak bergantung pada kamera atau model, jadi bisa diuji sendiri.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

ITEMS = ("mask", "glasses", "gloves")

# Nama kelas di dataset berbeda-beda. Semua nama di bawah dipetakan ke 4 kunci baku.
# Kelas "no_mask", "no_glove", dst. sengaja diabaikan (akurasinya rendah di banyak model).
ALIASES = {
    "mask": {"mask", "face-mask-medical", "face_mask", "face-mask", "masker", "medical-mask"},
    "glasses": {"goggles", "glasses", "safety-glasses", "safety_glasses", "safety glasses", "kacamata"},
    "gloves": {"gloves", "glove", "sarung tangan", "sarung-tangan", "sarung_tangan"},
    "person": {"person", "orang"},
}

DEFAULT_MIN_CONF = {"mask": 0.50, "glasses": 0.50, "gloves": 0.40}


def normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def build_class_map(names: dict) -> dict:
    """{id_kelas_model: kunci_baku}. Kelas yang tidak dikenal dibuang."""
    lookup = {}
    for key, aliases in ALIASES.items():
        for alias in aliases:
            lookup[normalize(alias)] = key
    return {int(i): lookup[normalize(n)] for i, n in names.items() if normalize(n) in lookup}


@dataclass
class Det:
    key: str          # mask / glasses / gloves
    conf: float
    box: tuple        # x1, y1, x2, y2


@dataclass
class Observation:
    """Pengamatan satu frame untuk satu orang.

    Mode sederhana: mask/glasses bool, gloves = jumlah sarung tangan yang terlihat.
    Mode smart (smart=True): mask/glasses/gloves_ok bernilai True (terlihat dipakai),
    False (area dilihat dan APD tidak ada) atau None (area tidak terlihat / tidak bisa dinilai).
    """
    mask: Optional[bool] = False
    glasses: Optional[bool] = False
    gloves: int = 0   # jumlah sarung tangan yang terlihat (mode sederhana)
    gloves_ok: Optional[bool] = None
    smart: bool = False


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _inside(pt, box):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def observe(dets, person_box=None, head_ratio=0.45, min_conf=None) -> Observation:
    """Ringkas deteksi satu frame menjadi pengamatan untuk satu orang.

    person_box=None -> mode "pos pemeriksaan": semua deteksi dianggap milik satu orang.
    person_box diisi -> hanya deteksi yang titik tengahnya di dalam kotak orang itu;
    masker/kacamata juga harus berada di bagian kepala (head_ratio teratas dari kotak orang).
    """
    thr = dict(DEFAULT_MIN_CONF)
    if min_conf:
        thr.update(min_conf)

    obs = Observation()
    glove_boxes = []
    for d in dets:
        if d.key not in ITEMS or d.conf < thr.get(d.key, 0.35):
            continue
        c = _center(d.box)
        if person_box is not None:
            if not _inside(c, person_box):
                continue
            if d.key in ("mask", "glasses"):
                head_limit = person_box[1] + head_ratio * (person_box[3] - person_box[1])
                if c[1] > head_limit:
                    continue
        if d.key == "mask":
            obs.mask = True
        elif d.key == "glasses":
            obs.glasses = True
        else:  # gloves: buang kotak ganda di tangan yang sama
            if any(_iou(d.box, g) > 0.5 for g in glove_boxes):
                continue
            glove_boxes.append(d.box)
            obs.gloves += 1
    return obs


class Smoother:
    """Menstabilkan hasil antar-frame supaya status tidak berkedip.

    Status per item:
      CHECKING : belum terkumpul `need` frame
      OK       : terlihat dipakai di minimal `need` dari `window` frame terakhir
      MISSING  : areanya terlihat cukup sering, tapi APD tidak ada
      UNSEEN   : areanya (wajah/tangan) sebagian besar tidak terlihat, jadi tidak bisa dinilai
    """

    def __init__(self, window=8, need=5, max_idle=45, required_gloves=1):
        self.window = window
        self.need = need
        self.max_idle = max_idle
        self.required_gloves = required_gloves
        self._hist = {}
        self._last_seen = {}
        self._frame = 0

    def tick(self):
        """Panggil sekali per frame, sebelum update()."""
        self._frame += 1
        stale = [p for p, f in self._last_seen.items() if self._frame - f > self.max_idle]
        for p in stale:
            self._hist.pop(p, None)
            self._last_seen.pop(p, None)

    def update(self, pid, obs: Observation) -> dict:
        h = self._hist.setdefault(pid, {k: deque(maxlen=self.window) for k in ITEMS})
        h["mask"].append(obs.mask)
        h["glasses"].append(obs.glasses)
        h["gloves"].append(obs.gloves_ok if obs.smart else obs.gloves >= self.required_gloves)
        self._last_seen[pid] = self._frame
        return {k: self._state(h[k]) for k in ITEMS}

    def _state(self, dq) -> str:
        if len(dq) < self.need:
            return "CHECKING"
        t = sum(1 for v in dq if v is True)
        f = sum(1 for v in dq if v is False)
        if t >= self.need:
            return "OK"
        if t + f >= self.need:
            return "MISSING"
        return "UNSEEN"


def overall(statuses: dict, required=ITEMS) -> str:
    """OK / VIOLATION / UNSEEN / CHECKING untuk item yang diwajibkan."""
    vals = [statuses[k] for k in required]
    if any(v == "MISSING" for v in vals):
        return "VIOLATION"
    if any(v == "UNSEEN" for v in vals):
        return "UNSEEN"
    if any(v == "CHECKING" for v in vals):
        return "CHECKING"
    return "OK"
