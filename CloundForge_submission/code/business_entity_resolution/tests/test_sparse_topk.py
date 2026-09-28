"""Sparse retrieval kernel vs a dense brute-force reference (src/ber/retrieval/sparse_topk.py)."""

import os
import sys
import unittest

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber.retrieval.sparse_topk import brute_force_topk, rank_rows, retrieve, score_block, topk_per_row  # noqa: E402


def random_problem(seed, n_q=40, n_t=300, n_tok=60, density_q=0.08, density_t=0.05):
    rng = np.random.default_rng(seed)
    # Integer-valued IDF weights create many exact ties, which the ordering must break by index.
    q = sp.random(n_q, n_tok, density=density_q, random_state=rng, format="csr",
                  data_rvs=lambda n: rng.integers(1, 4, n).astype(np.float32))
    inv = sp.random(n_tok, n_t, density=density_t, random_state=rng, format="csr",
                    data_rvs=lambda n: np.ones(n, dtype=np.float32))
    q.data = q.data.astype(np.float32)
    inv.data = inv.data.astype(np.float32)
    return q, inv


class TopKTest(unittest.TestCase):
    def assert_same(self, a, b):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_matches_brute_force(self):
        for seed in range(5):
            q, inv = random_problem(seed)
            for k in (None, 1, 3, 10, 1000):
                self.assert_same(topk_per_row(score_block(q, inv), k), brute_force_topk(q, inv, k))

    def test_chunked_equals_whole(self):
        q, inv = random_problem(7)
        rows, cols, scores, ranks = [], [], [], []
        for _, r, c, s, k in retrieve(q, inv, k=5, chunk_rows=7):
            rows.append(r); cols.append(c); scores.append(s); ranks.append(k)
        whole = topk_per_row(score_block(q, inv), 5)
        self.assert_same((np.concatenate(rows), np.concatenate(cols), np.concatenate(scores),
                          np.concatenate(ranks)), whole)

    def test_order_and_ties(self):
        m = sp.csr_matrix(np.array([[0, 2, 2, 5, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]], dtype=np.float32))
        rows, cols, scores, rank = topk_per_row(m)
        self.assertEqual(rows.tolist(), [0, 0, 0, 2, 2])
        self.assertEqual(cols.tolist(), [3, 1, 2, 0, 4])  # score desc, then column asc
        self.assertEqual(rank.tolist(), [0, 1, 2, 0, 1])
        rows, cols, _, _ = topk_per_row(m, 2)
        self.assertEqual(list(zip(rows.tolist(), cols.tolist())), [(0, 3), (0, 1), (2, 0), (2, 4)])

    def test_empty(self):
        m = sp.csr_matrix((3, 4), dtype=np.float32)
        self.assertEqual([len(x) for x in topk_per_row(m, 5)], [0, 0, 0, 0])
        order, rank = rank_rows(np.array([], np.int32), np.array([], np.int32), np.array([], np.float32), 3)
        self.assertEqual((len(order), len(rank)), (0, 0))

    def test_deterministic(self):
        q, inv = random_problem(3)
        self.assert_same(topk_per_row(score_block(q, inv), 4), topk_per_row(score_block(q, inv), 4))


if __name__ == "__main__":
    unittest.main()
