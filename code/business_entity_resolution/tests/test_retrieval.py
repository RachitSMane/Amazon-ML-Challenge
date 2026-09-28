"""Channels, unions and the streaming evaluator (src/ber/retrieval)."""

import os
import sys
import unittest

import numpy as np
import pyarrow as pa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber import tokens  # noqa: E402
from ber.retrieval.channels import C1, C2, K, ChannelRunner  # noqa: E402
from ber.retrieval.evaluate import RetrievalEvaluator, f05_from_recall  # noqa: E402
from ber.retrieval.union import union_candidates  # noqa: E402


def make_country_pool(target_idx, names, addresses, keys):
    cp = tokens.CountryPool(country="US", target_idx=np.asarray(target_idx, np.int32))
    cp.fields["name_fold"] = tokens.build_field_index(pa.array(names), "tokens")
    cp.fields["address_norm"] = tokens.build_field_index(pa.array(addresses), "tokens")
    cp.fields["name_key"] = tokens.build_field_index(pa.array(keys), "key")
    return cp


class ChannelTest(unittest.TestCase):
    def setUp(self):
        # Global pool positions 10..14 belong to this country.
        self.cp = make_country_pool(
            [10, 11, 12, 13, 14],
            ["acme group", "acme labs", "zen group", "zen labs group", "orbit group"],
            ["1 main street", "2 main street", "5 oak road", "", "1 main street"],
            ["acme", "acme labs", "zen", "labs zen", "orbit"],
        )
        q_names, q_addr, q_keys = pa.array(["acme group", "zen labs"]), pa.array(["1 main street", "9 pine"]), pa.array(["acme", "labs zen"])
        self.runner = ChannelRunner(self.cp, {
            "name_fold": tokens.query_matrix(q_names, "tokens", self.cp.fields["name_fold"]),
            "address_norm": tokens.query_matrix(q_addr, "tokens", self.cp.fields["address_norm"]),
            "name_key": tokens.query_matrix(q_keys, "key", self.cp.fields["name_key"]),
        })

    def pairs(self, channel):
        q, t, _, _ = self.runner.run(channel, 0, 2)
        return sorted(zip(q.tolist(), t.tolist()))

    def test_df_cap_suppresses_generic_tokens(self):
        # "group" has DF 4: with cap 3 it cannot retrieve; "acme" (DF 2) still does.
        self.assertEqual(self.pairs(C1(df_cap=3)), [(0, 10), (0, 11), (1, 11), (1, 12), (1, 13)])
        self.assertIn((0, 14), self.pairs(C1(df_cap=10)))  # uncapped "group" reaches orbit

    def test_scores_and_topk(self):
        q, t, score, rank = self.runner.run(C1(df_cap=10, top_k=1), 0, 2)
        self.assertEqual(list(zip(q.tolist(), t.tolist())), [(0, 10), (1, 13)])  # most shared IDF
        self.assertEqual(rank.tolist(), [0, 0])
        idf = dict(zip(self.cp.fields["name_fold"].vocab.to_pylist(), self.cp.fields["name_fold"].idf))
        self.assertAlmostEqual(float(score[1]), float(idf["zen"] + idf["labs"]), places=5)

    def test_key_channel_exact_only(self):
        self.assertEqual(self.pairs(K()), [(0, 10), (1, 13)])

    def test_address_channel_and_row_window(self):
        self.assertEqual(self.pairs(C2(df_cap=1000)), [(0, 10), (0, 11), (0, 14)])
        q, t, _, _ = self.runner.run(C2(df_cap=1000), 1, 2)
        self.assertEqual(len(q), 0)  # the second query's address shares nothing


class UnionTest(unittest.TestCase):
    parts = [
        (np.array([0, 0, 1], np.int32), np.array([5, 7, 5], np.int32), np.array([1.0, 3.0, 2.0], np.float32)),
        (np.array([0, 1], np.int32), np.array([5, 9], np.int32), np.array([4.0, 0.5], np.float32)),
    ]

    def test_dedupe_score_mask_rank(self):
        u = union_candidates(self.parts, 2)
        self.assertEqual(list(zip(u["q"].tolist(), u["t"].tolist())), [(0, 5), (0, 7), (1, 5), (1, 9)])
        self.assertEqual(u["score"].tolist(), [5.0, 3.0, 2.0, 0.5])
        self.assertEqual(u["mask"].tolist(), [3, 1, 1, 2])
        self.assertEqual(u["rank"].tolist(), [0, 1, 0, 1])

    def test_topk_and_determinism(self):
        u = union_candidates(self.parts, 2, top_k=1)
        self.assertEqual(list(zip(u["q"].tolist(), u["t"].tolist())), [(0, 5), (1, 5)])
        again = union_candidates(self.parts[::-1], 2, top_k=1)
        self.assertEqual(list(zip(again["q"].tolist(), again["t"].tolist())), [(0, 5), (1, 5)])

    def test_empty_part(self):
        empty = (np.array([], np.int32), np.array([], np.int32), np.array([], np.float32))
        u = union_candidates([empty, self.parts[1]], 2)
        self.assertEqual(u["mask"].tolist(), [2, 2])


class EvaluatorTest(unittest.TestCase):
    def setUp(self):
        # Query 0: truth {5, 7}; query 1: truth {9}; query 2: no truth.
        self.ev = RetrievalEvaluator(3, [0, 0, 1], [5, 7, 9])
        self.ev.add(np.array([0, 0, 0, 2]), np.array([5, 6, 8, 1]), np.array([0, 1, 2, 0]))
        self.ev.add(np.array([1]), np.array([4]), np.array([0]))

    def test_summary(self):
        s = self.ev.summary()
        self.assertEqual(s["candidates"]["total"], 5)
        self.assertEqual(s["pair_recall"], round(1 / 3, 4))
        self.assertEqual(s["s1_recall"], round((0.5 + 0) / 2, 4))
        self.assertEqual(s["pct_all_true_retrieved"], 0.0)
        self.assertEqual(s["pct_any_true_retrieved"], 50.0)
        f_q0 = 1.25 * 0.5 / (0.25 + 0.5)
        self.assertAlmostEqual(s["oracle_f05"], round((f_q0 + 0 + 1.0) / 3, 4))

    def test_topk_and_strata(self):
        s = self.ev.summary(k=1, query_strata={"q0": np.array([True, False, False])},
                            pair_strata={"first": np.array([True, False, False])})
        self.assertEqual(s["candidates"]["total"], 3)  # min(count, 1) per query
        self.assertEqual(s["by_pair_stratum"]["first"]["pair_recall"], 1.0)
        self.assertEqual(s["by_query_stratum"]["q0"]["mean_candidates"], 1.0)
        self.assertEqual(self.ev.retrieved(k=1).tolist(), [True, False, False])

    def test_f05(self):
        np.testing.assert_allclose(f05_from_recall([0, 1, 0.5]), [0, 1, 0.625 / 0.75])

    def test_duplicate_truth_rejected(self):
        with self.assertRaises(ValueError):
            RetrievalEvaluator(1, [0, 0], [3, 3])


if __name__ == "__main__":
    unittest.main()
