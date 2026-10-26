import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from make_dataset import split_chrono


class ChronoSplitTest(unittest.TestCase):
    def items(self, tag, n):
        return [(Path(f"{tag}_20261025_09{i:02d}00_000_{i:04d}.jpg"), []) for i in range(n)]

    def test_each_tag_split_in_time_blocks(self):
        items = self.items("kacamata_saja", 50) + self.items("masker_saja", 50)
        parts = split_chrono(items, [0.7, 0.2, 0.1])
        self.assertEqual(sum(len(v) for v in parts.values()), 100)
        for tag in ("kacamata_saja", "masker_saja"):
            tr = [i[0].name for i in parts["train"] if i[0].name.startswith(tag)]
            va = [i[0].name for i in parts["val"] if i[0].name.startswith(tag)]
            te = [i[0].name for i in parts["test"] if i[0].name.startswith(tag)]
            self.assertEqual((len(tr), len(va), len(te)), (35, 10, 5))
            self.assertLess(max(tr), min(va))          # train lebih awal dari val
            self.assertLess(max(va), min(te))          # val lebih awal dari test

    def test_small_group_still_has_val_and_test(self):
        parts = split_chrono(self.items("x", 4), [0.7, 0.2, 0.1])
        self.assertGreaterEqual(len(parts["val"]), 1)
        self.assertGreaterEqual(len(parts["test"]), 1)
        self.assertEqual(sum(len(v) for v in parts.values()), 4)


if __name__ == "__main__":
    unittest.main()
