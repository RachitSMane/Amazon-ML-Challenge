"""Chunked sparse scoring and per-row top-K selection.

``score_block(q_block, inverted)`` computes ``q_block @ inverted``: for every query row, the
sum of the IDF of the tokens it shares with each target (only targets sharing at least one
token get an entry, so nothing dense is ever built). ``topk_per_row`` then keeps the ``k``
best targets per query, ordered by score descending and target index ascending on ties,
which makes the result deterministic.
"""

import numpy as np
import scipy.sparse as sp


def score_block(q_block, inverted):
    """Sparse ``queries × targets`` IDF-overlap scores (float32 CSR)."""
    return (q_block @ inverted).tocsr()


def rank_rows(rows, cols, scores, n_rows):
    """Order entries by (row, score desc, col asc); return the permutation and in-row ranks.

    ``rows`` must be int arrays of equal length. The rank of an entry is its position
    within its row after ordering (0 = best).
    """
    order = np.lexsort((cols, -scores, rows))
    rows_sorted = rows[order]
    starts = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows_sorted, minlength=n_rows), out=starts[1:])
    rank = np.arange(len(order), dtype=np.int64) - starts[rows_sorted]
    return order, rank


def csr_to_pairs(m):
    """(rows int32, cols int32, data float32) of every stored entry of a CSR matrix."""
    rows = np.repeat(np.arange(m.shape[0], dtype=np.int32), np.diff(m.indptr))
    return rows, m.indices.astype(np.int32), m.data.astype(np.float32)


def topk_per_row(m, k=None):
    """Top-``k`` entries per row of a CSR score matrix (all entries if ``k`` is None).

    Returns ``(rows, cols, scores, rank)`` sorted by row, then rank.
    """
    rows, cols, scores = csr_to_pairs(m)
    order, rank = rank_rows(rows, cols, scores, m.shape[0])
    rows, cols, scores = rows[order], cols[order], scores[order]
    if k is not None:
        keep = rank < k
        rows, cols, scores, rank = rows[keep], cols[keep], scores[keep], rank[keep]
    return rows, cols, scores, rank.astype(np.int32)


def retrieve(q, inverted, k=None, chunk_rows=2000):
    """Yield ``(row_offset, rows, cols, scores, rank)`` for successive query chunks of ``q``."""
    for start in range(0, q.shape[0], chunk_rows):
        block = q[start:start + chunk_rows]
        rows, cols, scores, rank = topk_per_row(score_block(block, inverted), k)
        yield start, rows + start, cols, scores, rank


def brute_force_topk(q, inverted, k=None):
    """Dense reference implementation (tests only): same output contract as ``topk_per_row``."""
    dense = np.asarray((sp.csr_matrix(q).toarray() @ sp.csr_matrix(inverted).toarray()), dtype=np.float32)
    rows, cols = np.nonzero(dense)
    return topk_per_row(sp.csr_matrix((dense[rows, cols], (rows, cols)), shape=dense.shape), k)
