import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from auto_label import guess_glasses_box, yolo_line
from label_tool import boxes_to_yolo, expected_classes, yolo_to_boxes


class LabelToolTest(unittest.TestCase):
    def test_roundtrip(self):
        boxes = [(0, 100, 50, 300, 150), (1, 10, 10, 60, 40)]
        back = yolo_to_boxes(boxes_to_yolo(boxes, 640, 480), 640, 480)
        for a, b in zip(boxes, back):
            self.assertEqual(a[0], b[0])
            for u, v in zip(a[1:], b[1:]):
                self.assertAlmostEqual(u, v, delta=0.5)

    def test_drag_in_any_direction_and_clip(self):
        txt = boxes_to_yolo([(2, 300, 150, 100, 50), (0, -20, -20, 50, 50)], 640, 480)
        a, b = [list(map(float, l.split()[1:])) for l in txt.splitlines()]
        self.assertAlmostEqual(a[0], 200 / 640)           # tarik dari kanan-bawah ke kiri-atas tetap benar
        self.assertTrue(all(0 <= v <= 1 for v in a + b))  # keluar frame dipotong

    def test_tiny_box_dropped(self):
        self.assertEqual(boxes_to_yolo([(0, 10, 10, 12, 12)], 640, 480), "")

    def test_expected_from_filename(self):
        names = ["mask", "glasses", "gloves"]
        self.assertEqual(expected_classes("kacamata_hitam_masker_2026_0001.jpg", names), {0, 1})
        self.assertEqual(expected_classes("kacamata_saja_1.jpg", names), {1})
        self.assertEqual(expected_classes("masker_saja_1.jpg", names), {0})
        self.assertEqual(expected_classes("sarung_tangan_1.jpg", names), {2})
        self.assertEqual(expected_classes("tanpa_apd_1.jpg", names), set())

    def test_guess_glasses_box_shape(self):
        x1, y1, x2, y2 = guess_glasses_box((640, 180, 60), 1280, 720)
        self.assertAlmostEqual((x1 + x2) / 2, 640)
        self.assertAlmostEqual(x2 - x1, 120)          # sekitar 2x jarak mata
        self.assertLess(y2 - y1, x2 - x1)             # lebih lebar daripada tinggi
        self.assertTrue(yolo_line(1, (x1, y1, x2, y2), 1280, 720).startswith("1 0.5"))


if __name__ == "__main__":
    unittest.main()
