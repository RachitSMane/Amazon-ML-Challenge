"""Deterministic stratified folds and the development subset (src/ber/cv.py)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber import cv  # noqa: E402


def sample(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.array([f"S1-{i}" for i in rng.permutation(10 * n)[:n]], dtype=object)
    country = rng.choice(["US", "India"], n, p=[0.6, 0.4])
    bucket = cv.match_bucket(rng.choice([0, 1, 2, 3, 5, 11], n))
    return ids, np.array([f"{c}|{b}" for c, b in zip(country, bucket)], dtype=object)


class CvTest(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(cv.match_bucket([0, 1, 2, 3, 4, 11]).tolist(), ["0", "1", "2-3", "2-3", "4+", "4+"])

    def test_folds_reproducible_and_order_independent(self):
        ids, strata = sample()
        folds = cv.assign_folds(ids, strata)
        np.testing.assert_array_equal(folds, cv.assign_folds(ids, strata))
        perm = np.random.default_rng(1).permutation(len(ids))
        np.testing.assert_array_equal(cv.assign_folds(ids[perm], strata[perm]), folds[perm])

    def test_folds_balanced_per_stratum(self):
        ids, strata = sample()
        folds = cv.assign_folds(ids, strata, n_folds=5)
        for s in np.unique(strata):
            counts = np.bincount(folds[strata == s], minlength=5)
            self.assertLessEqual(counts.max() - counts.min(), 1, s)

    def test_seed_changes_assignment(self):
        ids, strata = sample()
        self.assertFalse(np.array_equal(cv.assign_folds(ids, strata, seed="a"), cv.assign_folds(ids, strata, seed="b")))

    def test_subset_exact_proportional_reproducible(self):
        ids, strata = sample()
        chosen = cv.stratified_subset(ids, strata, 1000)
        self.assertEqual(int(chosen.sum()), 1000)
        for s in np.unique(strata):
            share_all = (strata == s).mean()
            self.assertLessEqual(abs((strata[chosen] == s).mean() - share_all), 1 / 1000 + 1e-9, s)
        perm = np.random.default_rng(2).permutation(len(ids))
        np.testing.assert_array_equal(cv.stratified_subset(ids[perm], strata[perm], 1000), chosen[perm])

    def test_hash_is_stable(self):
        self.assertEqual(cv.stable_hash(["S1-1"])[0], cv.stable_hash(["S1-1"])[0])
        self.assertNotEqual(cv.stable_hash(["S1-1"])[0], cv.stable_hash(["S1-2"])[0])


if __name__ == "__main__":
    unittest.main()
