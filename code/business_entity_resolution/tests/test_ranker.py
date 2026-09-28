"""Stage-1 ranker (src/ber/ranker.py) and the Indic skeleton-key channel (retrieval/indic.py)."""

import os
import sys
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber import ranker, tokens  # noqa: E402
from ber.retrieval.channels import Channel, ChannelRunner  # noqa: E402
from ber.retrieval.indic import (INDIC_KEY_FIELD, build_indic_key_pools, name_skeleton,  # noqa: E402
                                 skeleton_key, skeleton_keys)

PARTS = [  # (q, t, score, rank) for channels "name", "addr"
    (np.array([0, 0, 1], np.int32), np.array([5, 7, 5], np.int32), np.array([1.0, 3.0, 2.0], np.float32),
     np.array([1, 0, 0], np.int32)),
    (np.array([0, 1], np.int32), np.array([5, 9], np.int32), np.array([4.0, 0.5], np.float32), np.array([0, 0], np.int32)),
]


def table(rows):
    cols = ["country", "name_script", "address_missing", "house_numbers", "unit_numbers", "postcode", "source"]
    data = dict(zip(cols, zip(*rows)))
    return pa.table({k: pa.array(list(v), pa.int8() if k == "source" else (pa.bool_() if k == "address_missing" else pa.string()))
                     for k, v in data.items()})


class MergeTest(unittest.TestCase):
    def test_merge(self):
        m = ranker.merge_channels(PARTS, 2)
        self.assertEqual(list(zip(m.q.tolist(), m.t.tolist())), [(0, 5), (0, 7), (1, 5), (1, 9)])
        np.testing.assert_array_equal(m.score, np.array([[1, 4], [3, np.nan], [2, np.nan], [np.nan, 0.5]], np.float32))
        np.testing.assert_array_equal(m.rank, np.array([[1, 0], [0, np.nan], [0, np.nan], [np.nan, 0]], np.float32))
        self.assertEqual(m.mask.tolist(), [3, 1, 1, 2])
        self.assertEqual(m.union_rank.tolist(), [0, 1, 0, 1])
        self.assertEqual(m.n_cand.tolist(), [2, 2, 2, 2])
        self.assertEqual(m.starts.tolist(), [0, 2, 4])

    def test_input_order_irrelevant(self):
        a, b = ranker.merge_channels(PARTS, 2), ranker.merge_channels([p[::1] for p in PARTS], 2)
        np.testing.assert_array_equal(a.t, b.t)


class FeatureTest(unittest.TestCase):
    def setUp(self):
        self.q = ranker.RecordArrays.from_table(table([
            ("US", "latin", False, "12", "4", "62704", 1), ("India", "latin", False, "", "", "", 1)]))
        self.t = ranker.RecordArrays.from_table(table([
            ("US", "latin", False, "0", "", "", 2)] * 5 + [
            ("US", "latin", False, "12", "5", "62704", 2), ("US", "latin", True, "", "", "", 3),
            ("US", "latin", False, "13", "4", "", 3), ("US", "devanagari", False, "", "", "", 2),
            ("India", "mixed", True, "", "", "", 3)]))
        n = 10
        self.t.extra.update({k: np.arange(n, dtype=np.float32) + 1 for k in ("name_ntok", "addr_ntok", "key_df",
                                                                             "tmass_name", "tmass_addr")})
        self.qstats = {"qmass_name": np.array([4.0, 2.0], np.float32), "qmass_addr": np.array([4.0, 1.0], np.float32),
                       "q_name_ntok": np.array([2, 1], np.float32), "q_addr_ntok": np.array([3, 2], np.float32),
                       "q_name_maxidf": np.array([5, 1], np.float32), "q_key_df": np.array([1, 0], np.float32)}
        self.m = ranker.merge_channels(PARTS, 2)
        self.q_global = np.array([0, 1])

    def features(self):
        return ranker.pair_features(self.m, ("name", "addr"), self.qstats, self.q_global, self.q, self.t)

    def test_shape_and_names(self):
        X = self.features()
        names = ranker.feature_names(("name", "addr"))
        self.assertEqual(X.shape, (4, len(names)))
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(X.dtype, np.float32)

    def col(self, name):
        return self.features()[:, ranker.feature_names(("name", "addr")).index(name)]

    def test_scores_and_ratios(self):
        np.testing.assert_allclose(self.col("s_name"), [1, 3, 2, np.nan])
        np.testing.assert_allclose(self.col("qfrac_name")[:2], [1 / 4, 3 / 4])
        np.testing.assert_allclose(self.col("tfrac_addr")[0], 4 / 6)  # target 5 has tmass 6
        np.testing.assert_allclose(self.col("s_sum"), [5, 3, 2, 0.5])
        np.testing.assert_allclose(self.col("s_rel"), [1, 3 / 5, 1, 0.25])
        np.testing.assert_allclose(self.col("n_channels"), [2, 1, 1, 1])

    def test_numbers(self):
        # Query 0 (house 12, unit 4, postcode 62704) vs targets 5 (12, 5, 62704) and 7 (13, 4, -).
        np.testing.assert_array_equal(self.col("house_eq")[:2], [1, 0])
        np.testing.assert_array_equal(self.col("house_conflict")[:2], [0, 1])
        np.testing.assert_array_equal(self.col("unit_eq")[:2], [0, 1])
        np.testing.assert_array_equal(self.col("unit_conflict")[:2], [1, 0])
        np.testing.assert_array_equal(self.col("post_eq")[:2], [1, 0])
        np.testing.assert_array_equal(self.col("post_conflict")[:2], [0, 0])  # target 7 has no postcode
        np.testing.assert_array_equal(self.col("house_eq")[2:], [0, 0])  # query 1 has no numbers

    def test_source_missing_country_script(self):
        np.testing.assert_array_equal(self.col("t_source"), [2, 3, 2, 3])
        np.testing.assert_array_equal(self.col("t_addr_missing"), [0, 0, 0, 1])
        np.testing.assert_array_equal(self.col("country"), [0, 0, 1, 1])
        np.testing.assert_array_equal(self.col("t_script"), [1, 1, 1, 2])

    def test_deterministic_and_label_free(self):
        X1, X2 = self.features(), self.features()
        np.testing.assert_array_equal(X1, X2)
        # pair_features takes no labels: labels computed afterwards cannot change the features.
        truth = np.sort(ranker.pair_keys([0], [7]))
        labels = ranker.labels_for(self.q_global[self.m.q], self.m.t, truth)
        self.assertEqual(labels.tolist(), [0, 1, 0, 0])
        np.testing.assert_array_equal(self.features(), X1)

    def test_first_value_hash(self):
        h = ranker.first_value_hash(pa.array(["12 14", "12", "", "13"]))
        self.assertEqual(h[0], h[1])
        self.assertEqual(h[2], 0)
        self.assertNotEqual(h[0], h[3])


class SamplingAndRankingTest(unittest.TestCase):
    def test_sampling_keeps_positives(self):
        labels = np.array([1, 0, 0, 0, 1, 0] * 1000, np.int8)
        urank = np.tile(np.arange(6), 1000)
        sel, w, hard = ranker.sample_negatives(labels, urank, np.random.default_rng(0), hard_rank=3, hard_rate=0.5, rest_rate=0.1)
        self.assertEqual(int(labels[sel].sum()), int(labels.sum()))
        self.assertTrue(np.all(w[labels[sel] == 1] == 1))
        self.assertTrue(set(np.round(w[labels[sel] == 0], 3)) <= {2.0, 10.0})
        sel2, _, _ = ranker.sample_negatives(labels, urank, np.random.default_rng(0), hard_rank=3, hard_rate=0.5, rest_rate=0.1)
        np.testing.assert_array_equal(sel, sel2)

    def test_rank_by_score_and_ties(self):
        q = np.array([0, 0, 0, 1, 1])
        t = np.array([9, 3, 5, 2, 1])
        rank = ranker.rank_by_score(q, t, np.array([0.5, 0.9, 0.5, 0.1, 0.1]), 2)
        self.assertEqual(rank.tolist(), [2, 0, 1, 1, 0])  # ties: smaller target index first

    def test_topk_pruning(self):
        rank = ranker.rank_by_score(np.zeros(5, int), np.arange(5), np.array([0.1, 0.5, 0.3, 0.9, 0.2]), 1)
        self.assertEqual(sorted(np.flatnonzero(rank < 2).tolist()), [1, 3])

    def test_training_s1_never_from_fold0_and_holdout_disjoint(self):
        split = pd.DataFrame({"entity_id": [f"S1-{i}" for i in range(2000)], "fold": np.arange(2000) % 5,
                              "stratum": np.where(np.arange(2000) % 2, "US|1", "India|2-3")})
        chosen = ranker.select_training_s1(split, 500)
        self.assertEqual(len(chosen), 500)
        self.assertFalse(set(chosen) & set(split.loc[split["fold"] == 0, "entity_id"]))
        hold = ranker.holdout_mask(chosen, 0.2)
        self.assertTrue(0.1 < hold.mean() < 0.3)
        np.testing.assert_array_equal(hold, ranker.holdout_mask(chosen, 0.2))

    def test_lightgbm_deterministic(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(3000, 4)).astype(np.float32)
        y = (X[:, 0] + 0.3 * rng.normal(size=3000) > 0).astype(int)
        w = np.ones(3000, np.float32)
        args = (X[:2500], y[:2500], w[:2500], X[2500:], y[2500:], w[2500:], ["a", "b", "c", "d"])
        b1, _ = ranker.train_lgbm(*args, params={"num_threads": 2, "min_data_in_leaf": 20}, rounds=50)
        b2, _ = ranker.train_lgbm(*args, params={"num_threads": 2, "min_data_in_leaf": 20}, rounds=50)
        np.testing.assert_array_equal(b1.predict(X[2500:]), b2.predict(X[2500:]))
        self.assertGreater(np.corrcoef(b1.predict(X[2500:]), X[2500:, 0])[0, 1], 0.5)


class IndicKeyTest(unittest.TestCase):
    def test_skeleton_key(self):
        self.assertEqual(skeleton_key("rm mrktnk rm"), "mrktnk rm")
        self.assertEqual(skeleton_keys(["राम मार्केटिंग", "Marketing Ram"]), ["mrktnk rm", "mrktnk rm"])
        self.assertEqual(name_skeleton("ग्लोबल इन्वेस्टमेंट"), name_skeleton("global investment"))

    def test_pool_indexes_only_indic_and_mixed(self):
        names = ["राम मार्केटिंग", "ram marketing", "royal हॉस्पिटैलिटी", "सन फूड्स"]
        pools = build_indic_key_pools(names, ["India", "India", "India", "US"],
                                      ["devanagari", "latin", "mixed", "devanagari"], ["India", "US"])
        self.assertEqual(pools["India"].target_idx.tolist(), [0, 2])
        self.assertEqual(pools["US"].target_idx.tolist(), [3])
        cp = pools["India"]
        q = tokens.query_matrix(pa.array(skeleton_keys(["Ram Marketing", "Sun Foods"]), pa.string()), "key", cp.fields[INDIC_KEY_FIELD])
        runner = ChannelRunner(cp, {INDIC_KEY_FIELD: q})
        qq, tt, _, _ = runner.run(Channel("indic_key", INDIC_KEY_FIELD), 0, 2)
        self.assertEqual(list(zip(qq.tolist(), tt.tolist())), [(0, 0)])  # "Sun Foods" is US-only


if __name__ == "__main__":
    unittest.main()
