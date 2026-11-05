import unittest
from scripts.benchmark_asr import distance, normalize


class MetricsTest(unittest.TestCase):
    def test_normalization_and_distance(self):
        self.assertEqual(normalize("Xin, CHÀO!"), "xin chào")
        self.assertEqual(distance("xin chào".split(), "xin bạn".split()), 1)
        self.assertEqual(distance("abc", "ac"), 1)


if __name__ == "__main__":
    unittest.main()
