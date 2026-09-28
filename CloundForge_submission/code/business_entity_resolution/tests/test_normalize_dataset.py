"""Data-safety checks of the normalized cache (work/cache/norm_v1) against the raw cache.

Skipped when the dataset or a fresh normalized cache is not available. Each source file is
checked on its own, so peak memory stays around one file (~1.5 GB).

Run from code/business_entity_resolution/:

    python -m unittest tests.test_normalize_dataset -v
"""

import os
import sys
import unittest

import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber import config, io  # noqa: E402
from ber.config import Config  # noqa: E402
from ber.normalize import NORMALIZED_SCHEMA  # noqa: E402
from ber.normalize import cache as norm_cache  # noqa: E402
from ber.normalize.text import SCRIPT_LABELS  # noqa: E402

try:
    CFG = Config.discover()
    STALE = [k for k in norm_cache.source_keys() if not norm_cache.is_norm_fresh(CFG, k)]
except FileNotFoundError:
    CFG, STALE = None, ["dataset not found"]

PASSTHROUGH = {"entity_id": "entity_id", "country": "country", "source": "source",
               "name_raw": "business_name", "address_raw": "business_address"}


@unittest.skipIf(STALE, f"normalized cache not available or stale: {STALE}")
class NormalizedCacheDatasetTest(unittest.TestCase):
    def test_every_source_file(self):
        ids = {}
        for split in config.SPLITS:
            for source in config.SOURCES:
                key = io.source_key(split, source)
                with self.subTest(key=key):
                    norm = pq.read_table(norm_cache.norm_file(CFG, key))
                    self.assertEqual(norm.schema, NORMALIZED_SCHEMA)
                    self.assertEqual(len(norm), config.EXPECTED_ROWS[(split, source)])
                    raw = pq.read_table(io.raw_cache_file(CFG, key), columns=list(PASSTHROUGH.values()))
                    for norm_col, raw_col in PASSTHROUGH.items():
                        # Same values in the same order: no row dropped, reordered or altered.
                        self.assertTrue(norm[norm_col].equals(raw[raw_col]), f"{key}.{norm_col}")
                    del raw
                    self.assertEqual(sum(c.null_count for c in norm.columns), 0)
                    self.assertEqual(pc.count_distinct(norm["entity_id"]).as_py(), len(norm))
                    scripts = set(pc.unique(norm["name_script"]).to_pylist()) | set(pc.unique(norm["address_script"]).to_pylist())
                    self.assertTrue(scripts <= set(SCRIPT_LABELS), scripts)
                    # address_missing <=> the normalized address is empty.
                    empty = pc.equal(norm["address_norm"], "")
                    self.assertTrue(norm["address_missing"].equals(empty), key)
                    if source == 1:
                        self.assertEqual(pc.sum(norm["address_missing"]).as_py(), 0)
                    ids[key] = norm["entity_id"]
                    del norm
        # Train and test stay separate: no ID appears in both splits.
        for source in config.SOURCES:
            test_ids, train_ids = ids[io.source_key("test", source)], ids[io.source_key("train", source)]
            self.assertFalse(pc.any(pc.is_in(test_ids, value_set=train_ids.combine_chunks())).as_py(), source)


if __name__ == "__main__":
    unittest.main()
