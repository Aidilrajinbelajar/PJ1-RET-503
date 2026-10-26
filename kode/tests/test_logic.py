import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppe_logic import Det, Observation, Smoother, build_class_map, observe, overall


class ClassMapTest(unittest.TestCase):
    def test_aliases(self):
        names = {0: "Gloves", 1: "Vest", 2: "goggles", 3: "helmet", 4: "mask", 5: "no_mask", 6: "Person"}
        self.assertEqual(build_class_map(names), {0: "gloves", 2: "glasses", 4: "mask", 6: "person"})

    def test_other_dataset_names(self):
        names = {0: "glove", 1: "Face-mask-medical", 2: "Glasses", 3: "no_glove"}
        self.assertEqual(build_class_map(names), {0: "gloves", 1: "mask", 2: "glasses"})


class ObserveTest(unittest.TestCase):
    def test_station_mode(self):
        dets = [
            Det("mask", 0.8, (100, 100, 150, 150)),
            Det("glasses", 0.7, (90, 60, 160, 90)),
            Det("gloves", 0.6, (10, 300, 60, 360)),
        ]
        o = observe(dets)
        self.assertTrue(o.mask)
        self.assertTrue(o.glasses)
        self.assertEqual(o.gloves, 1)

    def test_low_confidence_ignored(self):
        o = observe([Det("mask", 0.2, (0, 0, 10, 10))])
        self.assertFalse(o.mask)

    def test_duplicate_glove_boxes_counted_once(self):
        dets = [Det("gloves", 0.8, (10, 10, 60, 60)), Det("gloves", 0.5, (12, 12, 62, 62))]
        self.assertEqual(observe(dets).gloves, 1)

    def test_two_different_gloves(self):
        dets = [Det("gloves", 0.8, (10, 10, 60, 60)), Det("gloves", 0.8, (200, 10, 250, 60))]
        self.assertEqual(observe(dets).gloves, 2)

    def test_person_mode_head_region(self):
        person = (0, 0, 200, 400)
        # masker di perut orang itu (bukan di kepala) -> tidak dihitung
        low_mask = Det("mask", 0.9, (80, 250, 120, 290))
        self.assertFalse(observe([low_mask], person_box=person).mask)
        head_mask = Det("mask", 0.9, (80, 90, 120, 130))
        self.assertTrue(observe([head_mask], person_box=person).mask)

    def test_person_mode_separates_people(self):
        a, b = (0, 0, 200, 400), (300, 0, 500, 400)
        dets = [Det("mask", 0.9, (80, 90, 120, 130)), Det("gloves", 0.9, (350, 300, 400, 350))]
        oa, ob = observe(dets, a), observe(dets, b)
        self.assertTrue(oa.mask)
        self.assertEqual(oa.gloves, 0)
        self.assertFalse(ob.mask)
        self.assertEqual(ob.gloves, 1)


class SmootherTest(unittest.TestCase):
    def test_checking_then_ok(self):
        s = Smoother(window=8, need=5)
        full = Observation(mask=True, glasses=True, gloves=1)
        states = []
        for _ in range(6):
            s.tick()
            states.append(s.update(0, full)["mask"])
        self.assertEqual(states[:4], ["CHECKING"] * 4)
        self.assertEqual(states[4:], ["OK", "OK"])

    def test_missing_item(self):
        s = Smoother(window=8, need=5)
        obs = Observation(mask=True, glasses=False, gloves=1)
        for _ in range(8):
            s.tick()
            st = s.update(0, obs)
        self.assertEqual(st["mask"], "OK")
        self.assertEqual(st["glasses"], "MISSING")
        self.assertEqual(overall(st), "VIOLATION")
        self.assertEqual(overall(st, required=("mask", "gloves")), "OK")

    def test_flicker_is_smoothed(self):
        s = Smoother(window=8, need=5)
        for i in range(12):
            s.tick()
            st = s.update(0, Observation(mask=(i % 4 != 0), glasses=True, gloves=1))
        self.assertEqual(st["mask"], "OK")  # hilang 1 dari 4 frame, tetap OK

    def test_required_gloves_two(self):
        s = Smoother(window=8, need=5, required_gloves=2)
        for _ in range(8):
            s.tick()
            st = s.update(0, Observation(mask=True, glasses=True, gloves=1))
        self.assertEqual(st["gloves"], "MISSING")

    def test_stale_ids_purged(self):
        s = Smoother(max_idle=3)
        s.tick()
        s.update(7, Observation())
        for _ in range(5):
            s.tick()
        self.assertNotIn(7, s._hist)


class TriStateTest(unittest.TestCase):
    def _run(self, obs_list, **kw):
        s = Smoother(window=8, need=5, **kw)
        st = None
        for o in obs_list:
            s.tick()
            st = s.update(0, o)
        return st

    def test_unseen_when_area_never_visible(self):
        o = Observation(mask=None, glasses=None, gloves_ok=None, smart=True)
        st = self._run([o] * 8)
        self.assertEqual(st["mask"], "UNSEEN")
        self.assertEqual(overall(st), "UNSEEN")

    def test_seen_absent_is_missing(self):
        o = Observation(mask=False, glasses=True, gloves_ok=True, smart=True)
        st = self._run([o] * 8)
        self.assertEqual(st["mask"], "MISSING")
        self.assertEqual(st["glasses"], "OK")
        self.assertEqual(overall(st), "VIOLATION")

    def test_violation_beats_unseen(self):
        o = Observation(mask=False, glasses=True, gloves_ok=None, smart=True)
        st = self._run([o] * 8)
        self.assertEqual(st["gloves"], "UNSEEN")
        self.assertEqual(overall(st), "VIOLATION")

    def test_all_ok(self):
        o = Observation(mask=True, glasses=True, gloves_ok=True, smart=True)
        self.assertEqual(overall(self._run([o] * 8)), "OK")

    def test_occasional_unseen_frames_do_not_break_ok(self):
        seq = [Observation(mask=(None if i % 4 == 0 else True), glasses=True, gloves_ok=True, smart=True)
               for i in range(12)]
        self.assertEqual(self._run(seq)["mask"], "OK")

    def test_face_turns_away_after_ok(self):
        ok = Observation(mask=True, glasses=True, gloves_ok=True, smart=True)
        away = Observation(mask=None, glasses=None, gloves_ok=True, smart=True)
        st = self._run([ok] * 8 + [away] * 8)
        self.assertEqual(st["mask"], "UNSEEN")


if __name__ == "__main__":
    unittest.main()
