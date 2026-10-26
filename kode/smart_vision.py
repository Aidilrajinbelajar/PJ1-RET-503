"""Vision "pintar": pose untuk menemukan wajah & tangan, lalu deteksi APD di potongan gambar yang diperbesar.

Kenapa lebih akurat dari satu detektor di seluruh frame:
  * Kacamata, masker, dan sarung tangan kecil di frame penuh. Di potongan wajah/tangan ukurannya besar.
  * Sistem tahu mana yang "terlihat tapi tidak dipakai" (MISSING) dan mana yang "tidak terlihat" (UNSEEN),
    jadi tidak asal menuduh orang yang membelakangi kamera atau menyembunyikan tangan.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from geometry import eye_roi, face_anchor, face_roi, hand_rois, plausible_in_roi, plausible_on_face
from ppe_logic import DEFAULT_MIN_CONF, Det, Observation


@dataclass
class PersonResult:
    pid: int
    box: tuple
    obs: Observation
    face_roi: tuple = None          # area wajah yang dinilai (None = tidak bisa dinilai)
    face_reason: str = "not_visible"
    hands: list = field(default_factory=list)   # [(nama, roi, bersarung_tangan: bool)]
    dets: list = field(default_factory=list)    # deteksi APD yang DITERIMA (koordinat frame penuh)
    rejected: list = field(default_factory=list)  # [(Det, alasan)] deteksi yang ditolak filter posisi/ukuran
    anchor: tuple = None            # (ex, ey, jarak_mata) bila mata terlihat


class PoseGuided:
    def __init__(self, pose_model, ppe_model, class_map, pose_imgsz=416, crop_imgsz=320, conf=0.25,
                 max_people=3, min_conf=None, face_kp_thr=0.5, hand_kp_thr=0.4, geometry_filter=True, eye_crop=False):
        self.pose = pose_model
        self.ppe = ppe_model
        self.class_map = class_map
        self.pose_imgsz = pose_imgsz
        self.crop_imgsz = crop_imgsz
        self.conf = conf
        self.max_people = max_people
        self.thr = dict(DEFAULT_MIN_CONF)
        if min_conf:
            self.thr.update(min_conf)
        self.face_kp_thr = face_kp_thr
        self.hand_kp_thr = hand_kp_thr
        self.geometry_filter = geometry_filter
        self.eye_crop = eye_crop        # cek kacamata juga di potongan area mata saja

    def _detect_in(self, frame, roi, keys):
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        res = self.ppe.predict(crop, imgsz=self.crop_imgsz, conf=self.conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            key = self.class_map.get(int(b.cls[0]))
            if key not in keys:
                continue
            bx = b.xyxy[0].tolist()
            out.append(Det(key, float(b.conf[0]), (bx[0] + x1, bx[1] + y1, bx[2] + x1, bx[3] + y1)))
        return out

    def _has(self, dets, key):
        return any(d.key == key and d.conf >= self.thr[key] for d in dets)

    def analyze(self, frame) -> list:
        h, w = frame.shape[:2]
        res = self.pose.track(frame, imgsz=self.pose_imgsz, conf=self.conf, persist=True,
                              tracker="bytetrack.yaml", verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return []

        kxy = res.keypoints.xy.cpu().numpy()
        kcf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
        people = []
        for i, b in enumerate(res.boxes):
            box = tuple(float(v) for v in b.xyxy[0].tolist())
            pid = int(b.id[0]) if b.id is not None else i
            people.append((pid, box, kxy[i], kcf[i] if kcf is not None else [1.0] * len(kxy[i])))
        people.sort(key=lambda p: (p[1][2] - p[1][0]) * (p[1][3] - p[1][1]), reverse=True)

        results = []
        for pid, box, kps, kc in people[: self.max_people]:
            pr = PersonResult(pid=pid, box=box, obs=Observation(mask=None, glasses=None, gloves_ok=None, smart=True))

            roi, reason = face_roi(kps, kc, w, h, kp_thr=self.face_kp_thr)
            pr.face_roi, pr.face_reason = roi, reason
            if roi is not None:
                dets = self._detect_in(frame, roi, ("mask", "glasses"))
                anchor = face_anchor(kps, kc, self.face_kp_thr)
                pr.anchor = anchor
                if self.geometry_filter and anchor is not None:
                    kept = []
                    for d in dets:
                        ok, why = plausible_on_face(d.key, d.box, anchor)
                        (kept.append(d) if ok else pr.rejected.append((d, why)))
                    dets = kept
                if self.eye_crop and anchor is not None:
                    eroi = eye_roi(anchor, w, h)
                    if eroi is not None:
                        edets = self._detect_in(frame, eroi, ("glasses",))
                        if self.geometry_filter:
                            kept = []
                            for d in edets:
                                ok, why = plausible_on_face(d.key, d.box, anchor)
                                (kept.append(d) if ok else pr.rejected.append((d, why)))
                            edets = kept
                        dets = dets + edets
                pr.dets += dets
                pr.obs.mask = self._has(dets, "mask")
                pr.obs.glasses = self._has(dets, "glasses")

            states = []
            for name, hroi, _reason in hand_rois(kps, kc, w, h, kp_thr=self.hand_kp_thr):
                if hroi is None:
                    continue
                dets = self._detect_in(frame, hroi, ("gloves",))
                if self.geometry_filter:
                    kept = []
                    for d in dets:
                        ok, why = plausible_in_roi(d.box, hroi)
                        (kept.append(d) if ok else pr.rejected.append((d, why)))
                    dets = kept
                pr.dets += dets
                gloved = self._has(dets, "gloves")
                pr.hands.append((name, hroi, gloved))
                states.append(gloved)
            pr.obs.gloves_ok = all(states) if states else None
            results.append(pr)
        return results
