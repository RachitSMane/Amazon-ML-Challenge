"""Acceptance tests for the data layer on the real local dataset.

Skipped when no dataset is found (see ``ber.config.find_data_dir``). The reference numbers
come from the reconnaissance. The raw TSVs are the source of truth here (``use_cache=False``);
``test_parquet_cache_matches_tsv`` then checks the Parquet cache against them, and is skipped
if the cache has not been built. Takes about 1.5 minutes and ~6 GB of RAM.

Run from code/business_entity_resolution/:

    python -m unittest tests.test_io_dataset -v
"""

import os
import sys
import unittest

import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber import config, io  # noqa: E402
from ber.config import Config  # noqa: E402

try:
    DATA_DIR = config.find_data_dir()
except FileNotFoundError:
    DATA_DIR = None


def arrow(series):
    """Zero-copy Arrow view of a ``string[pyarrow]`` Series."""
    return pa.array(series)


def n_distinct(series):
    return pc.count_distinct(arrow(series)).as_py()


def all_in(values, value_set):
    return pc.all(pc.is_in(arrow(values), value_set=arrow(value_set))).as_py()


def none_in(values, value_set):
    return not pc.any(pc.is_in(arrow(values), value_set=arrow(value_set))).as_py()


@unittest.skipIf(DATA_DIR is None, "local dataset not found")
class RealDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config.discover(DATA_DIR)
        cls.s1 = {split: io.load_source(cls.cfg, split, 1, use_cache=False) for split in config.SPLITS}
        cls.targets = {split: io.load_targets(cls.cfg, split, use_cache=False) for split in config.SPLITS}
        cls.gt = io.load_ground_truth(cls.cfg, use_cache=False)

    def test_expected_files(self):
        self.assertEqual(config.missing_files(self.cfg.data_dir), [])

    def test_columns(self):
        expected = list(config.SOURCE_COLUMNS) + ["source"]
        for split in config.SPLITS:
            self.assertEqual(list(self.s1[split].columns), expected)
            self.assertEqual(list(self.targets[split].columns), expected)
        self.assertEqual(list(self.gt.s1.columns), ["source1_entity_id", "n_matches"])
        self.assertEqual(list(self.gt.pairs.columns),
                         ["source1_entity_id", "target_entity_id", "target_source"])

    def test_row_counts(self):
        for split in config.SPLITS:
            self.assertEqual(len(self.s1[split]), config.EXPECTED_ROWS[(split, 1)], split)
            counts = self.targets[split]["source"].value_counts().to_dict()
            self.assertEqual(counts, {2: config.EXPECTED_ROWS[(split, 2)],
                                      3: config.EXPECTED_ROWS[(split, 3)]}, split)

    def test_ids_unique(self):
        for split in config.SPLITS:
            s1, targets = self.s1[split], self.targets[split]
            self.assertEqual(n_distinct(s1["entity_id"]), len(s1), split)
            # S2 and S3 IDs are unique within and across the two files.
            self.assertEqual(n_distinct(targets["entity_id"]), len(targets), split)

    def test_no_empty_ids_or_countries(self):
        for split in config.SPLITS:
            for df in (self.s1[split], self.targets[split]):
                self.assertFalse((df["entity_id"] == "").any())
                self.assertFalse((df["country"] == "").any())

    def test_ground_truth_structure(self):
        s1, pairs = self.gt.s1, self.gt.pairs
        self.assertEqual(len(s1), config.EXPECTED_GROUND_TRUTH_ROWS)
        self.assertEqual(len(pairs), config.EXPECTED_GROUND_TRUTH_PAIRS)
        # Exactly one row per train S1 entity.
        self.assertEqual(n_distinct(s1["source1_entity_id"]), len(s1))
        self.assertTrue(all_in(self.s1["train"]["entity_id"], s1["source1_entity_id"]))
        self.assertTrue(all_in(s1["source1_entity_id"], self.s1["train"]["entity_id"]))
        # Every target is a valid train S2/S3 ID.
        self.assertTrue(all_in(pairs["target_entity_id"], self.targets["train"]["entity_id"]))
        # All target IDs are distinct: no duplicate inside a list and no target shared by two S1s.
        self.assertEqual(n_distinct(pairs["target_entity_id"]), len(pairs))
        self.assertEqual(int(s1["n_matches"].sum()), len(pairs))
        self.assertEqual(int(s1["n_matches"].max()), 11)
        self.assertAlmostEqual(float((s1["n_matches"] == 0).mean()), 0.056, delta=0.0005)
        self.assertAlmostEqual(float((s1["n_matches"] == 1).mean()), 0.054, delta=0.0005)
        self.assertAlmostEqual(float(s1["n_matches"].mean()), 3.46, delta=0.005)

    def test_train_test_separation(self):
        for label, train, test in (
            ("S1", self.s1["train"], self.s1["test"]),
            ("S2/S3", self.targets["train"], self.targets["test"]),
        ):
            self.assertTrue(none_in(test["entity_id"], train["entity_id"]), label)
        self.assertTrue(none_in(self.gt.pairs["target_entity_id"], self.targets["test"]["entity_id"]))

    def test_parquet_cache_matches_tsv(self):
        stale = [k for k in io.cache_keys() if not io.is_cache_fresh(self.cfg, k)]
        if stale:
            self.skipTest(f"Parquet cache missing or stale for {stale}; run `python -m ber.io build-cache`")
        manifest = io.read_manifest(self.cfg)
        for split in config.SPLITS:
            self.assertEqual(manifest["entries"][io.source_key(split, 1)]["rows"], config.EXPECTED_ROWS[(split, 1)])
            self.assertTrue(io.load_source(self.cfg, split, 1).equals(self.s1[split]), split)
            targets = io.load_targets(self.cfg, split)
            self.assertTrue(targets.equals(self.targets[split]), split)
            del targets
        gt = io.load_ground_truth(self.cfg)
        self.assertTrue(gt.s1.equals(self.gt.s1))
        self.assertTrue(gt.pairs.equals(self.gt.pairs))

    def test_countries(self):
        train = set(self.s1["train"]["country"].cat.categories) | set(self.targets["train"]["country"].cat.categories)
        test = set(self.s1["test"]["country"].cat.categories)
        self.assertNotIn("France", train)
        self.assertIn("France", test)


if __name__ == "__main__":
    unittest.main()
