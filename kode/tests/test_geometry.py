import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometry import L_EAR, L_ELB, L_EYE, L_SHO, L_WRI, NOSE, R_EAR, R_ELB, R_EYE, R_SHO, R_WRI, eye_roi, face_anchor, face_roi, hand_rois, plausible_in_roi, plausible_on_face


def pose(**over):
    """Pose 17 titik: orang menghadap kamera, tinggi frame 720."""
    k = np.zeros((17, 2))
    c = np.ones(17)
    k[NOSE] = (640, 200)
    k[L_EYE], k[R_EYE] = (670, 180), (610, 180)      # jarak mata 60 px
    k[L_EAR], k[R_EAR] = (700, 190), (580, 190)
    k[L_SHO], k[R_SHO] = (760, 330), (520, 330)
    k[L_ELB], k[R_ELB] = (800, 480), (480, 480)
    k[L_WRI], k[R_WRI] = (820, 620), (460, 620)      # lengan bawah sekitar 140 px
    for name, (xy, conf) in over.items():
        idx = globals()[name]
        k[idx] = xy
        c[idx] = conf
    return k, c


class FaceTest(unittest.TestCase):
    def test_face_visible(self):
        k, c = pose()
        roi, why = face_roi(k, c, 1280, 720)
        self.assertEqual(why, "ok")
        x1, y1, x2, y2 = roi
        self.assertAlmostEqual((x1 + x2) / 2, 640, delta=2)           # di tengah kedua mata
        self.assertAlmostEqual(x2 - x1, 4.2 * 60, delta=2)            # sisi = 4.2 x jarak mata
        self.assertLess(y1, 180)                                      # mencakup dahi
        self.assertGreater(y2, 240)                                   # mencakup dagu
        self.assertTrue(x1 <= k[NOSE][0] <= x2 and y1 <= k[NOSE][1] <= y2)

    def test_back_turned(self):
        k, c = pose(L_EYE=((670, 180), 0.1), R_EYE=((610, 180), 0.1))
        roi, why = face_roi(k, c, 1280, 720)
        self.assertIsNone(roi)
        self.assertEqual(why, "not_visible")

    def test_too_far(self):
        k, c = pose(L_EYE=((645, 180), 0.9), R_EYE=((635, 180), 0.9))   # jarak mata 10 px
        roi, why = face_roi(k, c, 1280, 720)
        self.assertIsNone(roi)
        self.assertEqual(why, "too_small")

    def test_clipped_at_frame_edge(self):
        k, c = pose(L_EYE=((40, 20), 0.9), R_EYE=((-20, 20), 0.9))
        roi, why = face_roi(k, c, 1280, 720)
        self.assertIsNotNone(roi)
        self.assertGreaterEqual(roi[0], 0)
        self.assertGreaterEqual(roi[1], 0)


class HandTest(unittest.TestCase):
    def test_both_hands(self):
        k, c = pose()
        hands = hand_rois(k, c, 1280, 720)
        self.assertEqual([h[0] for h in hands], ["kiri", "kanan"])
        for name, roi, why in hands:
            self.assertEqual(why, "ok")
            wrist = k[L_WRI] if name == "kiri" else k[R_WRI]
            self.assertTrue(roi[0] <= wrist[0] <= roi[2] and roi[1] <= wrist[1] <= roi[3])
            self.assertAlmostEqual(roi[2] - roi[0], 140, delta=6)

    def test_hand_center_shifted_past_wrist(self):
        k, c = pose()
        _, roi, _ = hand_rois(k, c, 1280, 720)[0]
        self.assertGreater((roi[1] + roi[3]) / 2, k[L_WRI][1])         # telapak di bawah pergelangan

    def test_hidden_wrist_is_skipped(self):
        k, c = pose(R_WRI=((460, 620), 0.05))
        self.assertEqual([h[0] for h in hand_rois(k, c, 1280, 720)], ["kiri"])

    def test_no_hands_visible(self):
        k, c = pose(L_WRI=((820, 620), 0.05), R_WRI=((460, 620), 0.05))
        self.assertEqual(hand_rois(k, c, 1280, 720), [])

    def test_elbow_missing_falls_back_to_shoulders(self):
        k, c = pose(L_ELB=((800, 480), 0.05))
        hands = hand_rois(k, c, 1280, 720)
        left = [h for h in hands if h[0] == "kiri"][0]
        self.assertEqual(left[2], "ok")

    def test_small_hand(self):
        k, c = pose(L_WRI=((800, 500), 0.9))        # lengan bawah 20 px -> tangan terlalu kecil
        hands = {h[0]: h for h in hand_rois(k, c, 1280, 720)}
        self.assertIsNone(hands["kiri"][1])
        self.assertEqual(hands["kiri"][2], "too_small")


class PlausibleTest(unittest.TestCase):
    """Mata di (640,180), jarak mata 60 px."""

    def setUp(self):
        k, c = pose()
        self.anchor = face_anchor(k, c)

    def test_anchor(self):
        ex, ey, d = self.anchor
        self.assertAlmostEqual(ex, 640)
        self.assertAlmostEqual(ey, 180)
        self.assertAlmostEqual(d, 60)

    def test_anchor_none_when_eyes_hidden(self):
        k, c = pose(L_EYE=((670, 180), 0.1))
        self.assertIsNone(face_anchor(k, c))

    def test_glasses_on_eyes_ok(self):
        self.assertEqual(plausible_on_face("glasses", (540, 150, 740, 215), self.anchor), (True, "ok"))

    def test_glasses_on_forehead_rejected(self):
        ok, why = plausible_on_face("glasses", (540, 40, 740, 100), self.anchor)
        self.assertFalse(ok)
        self.assertEqual(why, "posisi")

    def test_glasses_whole_head_rejected(self):
        ok, why = plausible_on_face("glasses", (480, 60, 800, 330), self.anchor)
        self.assertFalse(ok)

    def test_glasses_tiny_rejected(self):
        ok, why = plausible_on_face("glasses", (625, 170, 655, 190), self.anchor)
        self.assertEqual((ok, why), (False, "ukuran"))

    def test_mask_on_lower_face_ok(self):
        self.assertEqual(plausible_on_face("mask", (560, 215, 720, 320), self.anchor), (True, "ok"))

    def test_mask_above_eyes_rejected(self):
        ok, why = plausible_on_face("mask", (560, 60, 720, 150), self.anchor)
        self.assertEqual((ok, why), (False, "posisi"))

    def test_eye_roi_excludes_mask_area(self):
        roi = eye_roi(self.anchor, 1280, 720)
        self.assertTrue(roi[0] < 640 < roi[2] and roi[1] < 180 < roi[3])        # mata di dalam
        self.assertLess(roi[3], 180 + 0.95 * 60)                                  # bawah berhenti sebelum masker
        self.assertIsNone(eye_roi((20, 5, 4), 1280, 720))                         # terlalu kecil

    def test_glove_box_in_hand_roi(self):
        roi = (400, 560, 540, 700)
        self.assertEqual(plausible_in_roi((420, 580, 520, 690), roi), (True, "ok"))
        self.assertFalse(plausible_in_roi((100, 100, 160, 160), roi)[0])         # di luar area
        self.assertFalse(plausible_in_roi((300, 500, 700, 800), roi)[0])         # terlalu besar
        self.assertFalse(plausible_in_roi((450, 600, 460, 608), roi)[0])         # terlalu kecil


if __name__ == "__main__":
    unittest.main()
