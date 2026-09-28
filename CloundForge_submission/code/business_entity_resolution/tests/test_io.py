"""Fast unit tests for src/ber/config.py and src/ber/io.py on small synthetic TSV files.

Run from code/business_entity_resolution/:

    python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import hashlib

import pandas as pd
import pyarrow as pa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber import config, io  # noqa: E402
from ber.config import Config  # noqa: E402

HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"

FILES = {
    "train/train_source1.tsv": HEADER
    + "S1-1\tAcme Inc\t1 Main St, Springfield\tUS\n"
    + 'S1-2\t"Quoted" Traders\t"12, MG Road\tIndia\n'
    + "S1-3\tNo Match Co\t9 Elm St\tUS\n",
    "train/train_source2.tsv": HEADER
    + "S2-10\tACME INC.\t1 main street\tUS\n"
    + 'S2-11\tQuoted Traders "Pvt"\t\tIndia\n',
    "train/train_source3.tsv": HEADER + "S3-20\tAcme\t1 Main St\tUS\n",
    "train/train_ground_truth.tsv": "source1_entity_id\tmatched_entity_ids\n"
    + "S1-1\tS2-10, S3-20,\n"
    + "S1-2\tS2-11\n"
    + "S1-3\t\n",
    "test/test_source1.tsv": HEADER + "S1-7\tBoulangerie Martin SARL\t3 rue de la Paix\tFrance\n",
    "test/test_source2.tsv": HEADER + "S2-70\tBoulangerie Martin\t3 r. de la Paix\tFrance\n",
    "test/test_source3.tsv": HEADER + "S3-80\tMartin\t\tFrance\n",
}


def write_dataset(root, files=FILES):
    """Write the synthetic dataset under ``root`` and return ``root``."""
    for rel, text in files.items():
        path = Path(root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    return Path(root)


class DatasetDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = write_dataset(Path(self._tmp.name) / "dataset")
        self.work = Path(self._tmp.name) / "work"
        self.cfg = Config.discover(self.root, self.work)

    def tearDown(self):
        self._tmp.cleanup()

    def rewrite(self, rel, text):
        (self.root / rel).write_text(text, encoding="utf-8", newline="\n")


class FindDataDirTest(DatasetDirTestCase):
    def test_explicit_dir(self):
        self.assertEqual(config.find_data_dir(self.root), self.root.resolve())

    def test_explicit_incomplete_dir_raises(self):
        os.remove(self.root / "test/test_source3.tsv")
        with self.assertRaisesRegex(FileNotFoundError, "test_source3"):
            config.find_data_dir(self.root)

    def test_env_var(self):
        with mock.patch.dict(os.environ, {config.DATA_DIR_ENV: str(self.root)}):
            self.assertEqual(config.find_data_dir(), self.root.resolve())

    def test_expected_file_names(self):
        self.assertEqual(
            sorted(config.EXPECTED_FILES),
            sorted([
                "train/train_source1.tsv", "train/train_source2.tsv", "train/train_source3.tsv",
                "train/train_ground_truth.tsv",
                "test/test_source1.tsv", "test/test_source2.tsv", "test/test_source3.tsv",
            ]),
        )


class ConfigTest(DatasetDirTestCase):
    def test_paths(self):
        self.assertEqual(self.cfg.work_dir, self.work.resolve())
        self.assertEqual(self.cfg.raw_cache_dir, self.work.resolve() / "cache" / f"raw_v{config.RAW_CACHE_VERSION}")
        self.assertEqual(self.cfg.source_path("test", 2), self.root.resolve() / "test/test_source2.tsv")
        self.assertEqual(self.cfg.ground_truth_path, self.root.resolve() / "train/train_ground_truth.tsv")

    def test_default_work_dir_is_repo_relative(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(config.WORK_DIR_ENV, None)
            self.assertEqual(config.find_work_dir(), config.REPO_ROOT / "work")

    def test_work_dir_env_var(self):
        with mock.patch.dict(os.environ, {config.WORK_DIR_ENV: str(self.work)}):
            self.assertEqual(Config.discover(self.root).work_dir, self.work.resolve())

    def test_work_dir_inside_dataset_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            Config.discover(self.root, self.root / "work")

    def test_dataset_inside_work_dir_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            Config.discover(self.root, self.root.parent)

    def test_bad_block_size_rejected(self):
        for bad in (0, -1, 1024, 2 << 30, 1.5):
            with self.assertRaisesRegex(ValueError, "block_size"):
                Config.discover(self.root, self.work, block_size=bad)

    def test_incomplete_dataset_rejected_by_validate(self):
        os.remove(self.root / "train/train_ground_truth.tsv")
        with self.assertRaises(FileNotFoundError):
            Config(data_dir=self.root, work_dir=self.work).validate()


def file_hashes(root):
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(root).rglob("*.tsv"))}


class ParquetCacheTest(DatasetDirTestCase):
    def test_round_trip_matches_tsv(self):
        manifest = io.build_raw_cache(self.cfg)
        self.assertEqual(sorted(manifest["entries"]), sorted(io.cache_keys()))
        for split in config.SPLITS:
            for source in config.SOURCES:
                self.assertTrue(io.is_cache_fresh(self.cfg, io.source_key(split, source)))
                cached = io.load_source(self.cfg, split, source)
                raw = io.load_source(self.cfg, split, source, use_cache=False)
                pd.testing.assert_frame_equal(cached, raw)
            pd.testing.assert_frame_equal(io.load_targets(self.cfg, split),
                                          io.load_targets(self.cfg, split, use_cache=False))
        cached, raw = io.load_ground_truth(self.cfg), io.load_ground_truth(self.cfg, use_cache=False)
        pd.testing.assert_frame_equal(cached.s1, raw.s1)
        pd.testing.assert_frame_equal(cached.pairs, raw.pairs)
        self.assertEqual(manifest["entries"]["train_source1"]["rows"], 3)
        self.assertEqual(manifest["entries"][io.GROUND_TRUTH_KEY]["rows"], 3)

    def test_cache_lives_under_work_dir_only(self):
        before = set(self.root.rglob("*"))
        io.build_raw_cache(self.cfg)
        self.assertEqual(set(self.root.rglob("*")), before)
        self.assertEqual(len(list(self.cfg.raw_cache_dir.glob("*.parquet"))), 8)
        self.assertEqual(list(self.cfg.raw_cache_dir.glob("*.tmp")), [])

    def test_raw_files_unchanged(self):
        before = file_hashes(self.root)
        io.build_raw_cache(self.cfg, force=True)
        self.assertEqual(file_hashes(self.root), before)

    def test_fresh_entries_are_skipped(self):
        io.build_raw_cache(self.cfg)
        messages = []
        io.build_raw_cache(self.cfg, log=messages.append)
        self.assertTrue(all("skipped" in m for m in messages), messages)
        self.assertEqual(len(messages), len(io.cache_keys()))

    def test_stale_cache_is_not_used(self):
        io.build_raw_cache(self.cfg)
        self.rewrite("train/train_source3.tsv", HEADER + "S3-20\tAcme\t1 Main St\tUS\nS3-21\tNew\tAddr\tUS\n")
        self.assertFalse(io.is_cache_fresh(self.cfg, "train_source3"))
        self.assertEqual(len(io.load_source(self.cfg, "train", 3)), 2)  # falls back to the TSV
        io.build_raw_cache(self.cfg)
        self.assertTrue(io.is_cache_fresh(self.cfg, "train_source3"))
        self.assertEqual(len(io.load_source(self.cfg, "train", 3)), 2)

    def test_chunks_from_cache(self):
        rows = "".join(f"S2-{i}\tName {i}\tAddr {i}\tUS\n" for i in range(500))
        self.rewrite("train/train_source2.tsv", HEADER + rows)
        io.build_raw_cache(self.cfg)
        chunks = list(io.iter_source_chunks(self.cfg, "train", 2, block_size=1024))
        self.assertGreater(len(chunks), 1)
        joined = pd.concat([c.astype({"country": "string[pyarrow]"}) for c in chunks], ignore_index=True)
        full = io.load_source(self.cfg, "train", 2, use_cache=False)
        pd.testing.assert_frame_equal(joined, full.astype({"country": "string[pyarrow]"}))

    def test_cli(self):
        with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            code = io.main(["build-cache", "--data-dir", str(self.root), "--work-dir", str(self.work)])
        self.assertEqual(code, 0)
        self.assertTrue(all(io.is_cache_fresh(self.cfg, k) for k in io.cache_keys()))


class LoadSourceTest(DatasetDirTestCase):
    def test_columns_and_dtypes(self):
        df = io.load_source(self.cfg, "train", 1)
        self.assertEqual(list(df.columns), list(config.SOURCE_COLUMNS) + ["source"])
        for col in config.SOURCE_COLUMNS[:3]:
            self.assertEqual(df[col].dtype, pd.StringDtype("pyarrow"), col)
        self.assertIsInstance(df["country"].dtype, pd.CategoricalDtype)
        self.assertEqual(df["source"].dtype, "int8")
        self.assertEqual(df["source"].tolist(), [1, 1, 1])
        self.assertEqual(df["entity_id"].tolist(), ["S1-1", "S1-2", "S1-3"])

    def test_quotes_are_literal(self):
        df = io.load_source(self.cfg, "train", 1)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.loc[1, "business_name"], '"Quoted" Traders')
        self.assertEqual(df.loc[1, "business_address"], '"12, MG Road')
        self.assertEqual(df.loc[1, "country"], "India")

    def test_empty_address_is_empty_string(self):
        df = io.load_source(self.cfg, "train", 2)
        self.assertEqual(df.loc[1, "business_address"], "")
        self.assertFalse(df.isna().any().any())

    def test_wrong_prefix_raises(self):
        self.rewrite("train/train_source2.tsv", HEADER + "S3-99\tX\tY\tUS\n")
        with self.assertRaisesRegex(ValueError, "S2-"):
            io.load_source(self.cfg, "train", 2)

    def test_bad_header_raises(self):
        self.rewrite("train/train_source3.tsv", "id\tname\taddress\tcountry\nS3-1\tA\tB\tUS\n")
        with self.assertRaisesRegex(ValueError, "header"):
            io.load_source(self.cfg, "train", 3)

    def test_wrong_field_count_raises(self):
        self.rewrite("train/train_source3.tsv", HEADER + "S3-1\tA\tB\n")
        with self.assertRaises(pa.ArrowInvalid):
            io.load_source(self.cfg, "train", 3)

    def test_chunked_equals_full(self):
        rows = "".join(f"S2-{i}\tName {i}\tAddr {i}\t{'US' if i % 2 else 'India'}\n" for i in range(500))
        self.rewrite("train/train_source2.tsv", HEADER + rows)
        full = io.load_source(self.cfg, "train", 2)
        chunks = list(io.iter_source_chunks(self.cfg, "train", 2, block_size=1024))
        self.assertGreater(len(chunks), 1)
        joined = pd.concat([c.astype({"country": "string[pyarrow]"}) for c in chunks], ignore_index=True)
        pd.testing.assert_frame_equal(joined, full.astype({"country": "string[pyarrow]"}))

    def test_load_targets(self):
        df = io.load_targets(self.cfg, "train")
        self.assertEqual(df["entity_id"].tolist(), ["S2-10", "S2-11", "S3-20"])
        self.assertEqual(df["source"].tolist(), [2, 2, 3])
        self.assertEqual(sorted(df["country"].cat.categories), ["India", "US"])

    def test_test_split(self):
        df = io.load_source(self.cfg, "test", 1)
        self.assertEqual(df["country"].tolist(), ["France"])


class GroundTruthTest(DatasetDirTestCase):
    def test_structure(self):
        gt = io.load_ground_truth(self.cfg)
        self.assertEqual(gt.s1["source1_entity_id"].tolist(), ["S1-1", "S1-2", "S1-3"])
        self.assertEqual(gt.s1["n_matches"].tolist(), [2, 1, 0])
        self.assertEqual(gt.s1["n_matches"].dtype, "int16")
        self.assertEqual(gt.pairs["source1_entity_id"].tolist(), ["S1-1", "S1-1", "S1-2"])
        self.assertEqual(gt.pairs["target_entity_id"].tolist(), ["S2-10", "S3-20", "S2-11"])
        self.assertEqual(gt.pairs["target_source"].tolist(), [2, 3, 2])
        self.assertEqual(gt.pairs["target_source"].dtype, "int8")

    def test_bad_target_prefix_raises(self):
        self.rewrite("train/train_ground_truth.tsv",
                     "source1_entity_id\tmatched_entity_ids\nS1-1\tS1-2\n")
        with self.assertRaisesRegex(ValueError, "S2-"):
            io.load_ground_truth(self.cfg)

    def test_bad_s1_prefix_raises(self):
        self.rewrite("train/train_ground_truth.tsv",
                     "source1_entity_id\tmatched_entity_ids\nS2-1\tS2-10\n")
        with self.assertRaisesRegex(ValueError, "S1-"):
            io.load_ground_truth(self.cfg)


if __name__ == "__main__":
    unittest.main()
