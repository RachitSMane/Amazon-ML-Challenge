"""Tokenization, vocabularies, DF/IDF, DF caps and country-partitioned pools (src/ber/tokens.py)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from ber import io, tokens  # noqa: E402
from ber.config import Config  # noqa: E402
from ber.normalize import cache as norm_cache  # noqa: E402
from test_io import FILES, HEADER, write_dataset  # noqa: E402


class TokenizeTest(unittest.TestCase):
    def test_tokens_mode(self):
        doc, toks = tokens.tokenize(pa.array(["a b", "", "c a a"]), "tokens")
        self.assertEqual(doc.tolist(), [0, 0, 2, 2, 2])
        self.assertEqual(toks.to_pylist(), ["a", "b", "c", "a", "a"])

    def test_key_mode(self):
        doc, toks = tokens.tokenize(pa.array(["acme", "", "b a"]), "key")
        self.assertEqual((doc.tolist(), toks.to_pylist()), ([0, 2], ["acme", "b a"]))

    def test_empty_untyped_array(self):
        idx = tokens.build_field_index(pa.array([]), "tokens")
        self.assertEqual((idx.n_tokens, idx.n_docs, idx.inverted.shape), (0, 0, (0, 0)))
        q = tokens.query_matrix(pa.array(["a b"]), "tokens", idx)
        self.assertEqual((q.shape, q.nnz), ((1, 0), 0))

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            tokens.tokenize(pa.array(["a"]), "chars")


class FieldIndexTest(unittest.TestCase):
    def setUp(self):
        self.idx = tokens.build_field_index(pa.array(["b a", "a a c", "", "a"]), "tokens")

    def test_vocab_sorted_and_df(self):
        self.assertEqual(self.idx.vocab.to_pylist(), ["a", "b", "c"])
        self.assertEqual(self.idx.df.tolist(), [3, 1, 1])  # "a" twice in doc 1 counts once
        np.testing.assert_allclose(self.idx.idf, np.log(4 / np.array([3, 1, 1])), rtol=1e-6)

    def test_inverted_index(self):
        dense = self.idx.inverted.toarray()
        self.assertEqual(dense.tolist(), [[1, 1, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
        self.assertEqual(self.idx.inverted.indices.dtype, np.int32)

    def test_row_order_does_not_change_vocab(self):
        other = tokens.build_field_index(pa.array(["a", "", "a a c", "b a"]), "tokens")
        self.assertEqual(other.vocab.to_pylist(), self.idx.vocab.to_pylist())
        self.assertEqual(other.df.tolist(), self.idx.df.tolist())

    def test_query_matrix(self):
        q = tokens.query_matrix(pa.array(["a zzz", "c c b", "", "zzz"]), "tokens", self.idx)
        self.assertEqual(q.shape, (4, 3))
        dense = q.toarray()
        np.testing.assert_allclose(dense[0], [self.idx.idf[0], 0, 0], rtol=1e-6)  # OOV dropped
        np.testing.assert_allclose(dense[1], [0, self.idx.idf[1], self.idx.idf[2]], rtol=1e-6)  # dup once
        self.assertEqual(dense[2].sum() + dense[3].sum(), 0)

    def test_cap(self):
        q = tokens.query_matrix(pa.array(["a b", "a", "c"]), "tokens", self.idx)
        capped = tokens.cap_query_matrix(q, self.idx.df, 2)
        self.assertEqual((capped.toarray() > 0).tolist(), [[False, True, False], [False, False, False], [False, False, True]])
        self.assertIs(tokens.cap_query_matrix(q, self.idx.df, None), q)
        self.assertEqual(q.nnz, 4)  # the uncapped matrix is untouched

    def test_df_summary(self):
        summary = tokens.df_summary(self.idx, top=2)
        self.assertEqual(summary["n_tokens"], 3)
        self.assertEqual(summary["top_tokens"], [["a", 3], ["b", 1]])


POOL_FILES = dict(FILES)
POOL_FILES["train/train_source2.tsv"] = (
    HEADER + "S2-10\tACME INC.\t1 main street\tUS\n" + "S2-11\tSharma Traders\t12 MG Road\tIndia\n"
    + "S2-12\tAcme Labs\t9 Elm St\tUS\n"
)
POOL_FILES["train/train_source3.tsv"] = HEADER + "S3-20\tAcme\t1 Main St\tUS\n" + "S3-21\tSharma\t\tIndia\n"


class TargetPoolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = write_dataset(Path(cls._tmp.name) / "dataset", POOL_FILES)
        cls.cfg = Config.discover(root, Path(cls._tmp.name) / "work")
        io.build_raw_cache(cls.cfg)
        norm_cache.build_norm_cache(cls.cfg, workers=1)
        cls.pool = tokens.build_target_pool(cls.cfg, "train")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_order_and_sources(self):
        self.assertEqual(self.pool.entity_id.to_pylist(), ["S2-10", "S2-11", "S2-12", "S3-20", "S3-21"])
        self.assertEqual(self.pool.source.tolist(), [2, 2, 2, 3, 3])
        self.assertEqual(self.pool.address_missing.tolist(), [False, False, False, False, True])

    def test_country_partition(self):
        self.assertEqual(sorted(self.pool.countries), ["India", "US"])
        self.assertEqual(self.pool.countries["US"].target_idx.tolist(), [0, 2, 3])
        self.assertEqual(self.pool.countries["India"].target_idx.tolist(), [1, 4])
        us_name = self.pool.countries["US"].fields["name_fold"]
        self.assertEqual(us_name.n_docs, 3)
        self.assertEqual(dict(zip(us_name.vocab.to_pylist(), us_name.df.tolist())),
                         {"acme": 3, "inc": 1, "labs": 1})
        india_addr = self.pool.countries["India"].fields["address_norm"]
        self.assertEqual(india_addr.n_docs, 2)
        self.assertEqual(india_addr.inverted.shape[1], 2)
        key = self.pool.countries["US"].fields["name_key"]
        self.assertEqual(key.vocab.to_pylist(), ["acme", "acme labs"])  # "acme inc" -> key "acme"


if __name__ == "__main__":
    unittest.main()
