import sys
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber import matcher, ranker  # noqa: E402
from evaluation import f_beta  # noqa: E402


class MacroF05(unittest.TestCase):
    def test_matches_official_rules(self):
        cases = [(0, 0, 0), (1, 0, 0), (0, 2, 0), (3, 2, 2), (2, 2, 1), (1, 3, 0), (4, 4, 4)]
        n_pred, n_true, n_hit = map(np.array, zip(*cases))
        expected = np.mean([f_beta(p, t, h) for p, t, h in cases])
        self.assertAlmostEqual(matcher.macro_f05(n_pred, n_true, n_hit), expected, places=12)


class Decision(unittest.TestCase):
    def test_one_s1_per_target(self):
        q = np.array([0, 1, 1, 2])
        t = np.array([7, 7, 8, 8])
        p = np.array([0.9, 0.95, 0.6, 0.6])
        self.assertEqual(matcher.best_s1_per_target(q, t, p).tolist(), [False, True, True, False])
        self.assertEqual(matcher.decide(q, t, p, 0.7).tolist(), [False, True, False, False])
        self.assertEqual(matcher.decide(q, t, p, 0.7, one_to_one=False).tolist(), [True, True, False, False])

    def test_tune_threshold_prefers_precision(self):
        q = np.array([0, 0, 1])
        t = np.array([1, 2, 3])
        p = np.array([0.9, 0.4, 0.3])
        y = np.array([1, 0, 0])
        thr, best, _ = matcher.tune_threshold(q, t, p, y, 2, np.array([1, 0]))
        self.assertEqual(best["macro_f05"], 1.0)
        self.assertTrue(0.4 < thr <= 0.9)


class Candidates(unittest.TestCase):
    def test_cap_merged_keeps_prefix_and_offsets(self):
        parts = [(np.array([0, 0, 0, 1, 1]), np.array([5, 6, 7, 5, 8]), np.array([3., 2., 1., 5., 4.], np.float32),
                  np.array([0, 1, 2, 0, 1], np.float32))]
        m = ranker.merge_channels(parts, 2)
        c, rows = matcher.cap_merged(m, 2)
        self.assertEqual(c.union_rank.tolist(), [0, 1, 0, 1])
        self.assertEqual(c.starts.tolist(), [0, 2, 4])
        np.testing.assert_array_equal(c.union_score[c.starts[c.q]], m.union_score[m.starts[m.q]][rows])


class StringFeatures(unittest.TestCase):
    def test_empty_strings_are_nan(self):
        names = ranker.feature_names(("name",))
        n = 2
        X1 = np.zeros((n, len(names)), np.float32)
        qs = {c: pa.array(["acme corp", ""]) for c in matcher.STRING_COLUMNS}
        ts = {c: pa.array(["acme corporation", "x"]) for c in matcher.STRING_COLUMNS}
        X = matcher.stage2_features(X1, names, np.array([1.0, 0.5], np.float32), np.array([0, 1], np.float32),
                                    np.array([0, 0]), 1, qs, ts)
        out = matcher.stage2_feature_names(names)
        self.assertEqual(X.shape, (n, len(out)))
        self.assertNotIn("country", out)
        j = out.index("name_ratio")
        self.assertTrue(0 < X[0, j] < 100)
        self.assertTrue(np.isnan(X[1, j]))
        self.assertEqual(X[0, out.index("s1_gap12")], 0.5)


if __name__ == "__main__":
    unittest.main()
