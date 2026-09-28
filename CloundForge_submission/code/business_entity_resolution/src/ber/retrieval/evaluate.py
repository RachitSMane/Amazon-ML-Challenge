"""Streaming evaluation of candidate sets against the ground truth.

Candidates arrive chunk by chunk (``add``) and are never stored: the evaluator keeps one
counter per query and, for every true pair, the best rank at which it was retrieved. Any
union-level top-K can then be scored afterwards (``summary(k=...)``) from the same pass.

Definitions (macro over queries, like the challenge metric):

* pair recall: retrieved true pairs / all true pairs;
* S1 recall: mean over queries with >= 1 true match of (retrieved / true);
* oracle F0.5: F0.5 of a perfect matcher restricted to the candidates, i.e. predicting
  exactly the retrieved true targets: 1.0 for a query with no true match (predict empty),
  otherwise ``1.25 R / (0.25 + R)`` with R = its recall (0 when nothing is retrieved).
"""

import numpy as np

NOT_FOUND = np.iinfo(np.int32).max


def f05_from_recall(recall):
    """Oracle F0.5 (precision 1) for an array of recalls."""
    recall = np.asarray(recall, dtype=np.float64)
    return np.where(recall > 0, 1.25 * recall / (0.25 + np.where(recall > 0, recall, 1)), 0.0)


def _quantiles(counts):
    if not len(counts):
        return {}
    q = np.quantile(counts, [0.5, 0.9, 0.95, 0.99])
    return {"mean": round(float(counts.mean()), 2), "median": float(q[0]), "p90": float(q[1]),
            "p95": float(q[2]), "p99": float(q[3]), "max": int(counts.max()), "total": int(counts.sum())}


class RetrievalEvaluator:
    """Accumulates candidate counts per query and the best rank of every true pair."""

    def __init__(self, n_queries, truth_q, truth_t):
        truth_q = np.asarray(truth_q, dtype=np.int64)
        truth_t = np.asarray(truth_t, dtype=np.int64)
        keys = (truth_q << 32) | truth_t
        order = np.argsort(keys, kind="stable")
        self.truth_keys = keys[order]
        if len(self.truth_keys) and (np.diff(self.truth_keys) == 0).any():
            raise ValueError("duplicate true pairs")
        self.truth_q = truth_q[order]
        self.truth_order = order  # position in sorted arrays -> original pair index
        self.n_queries = n_queries
        self.n_true = np.bincount(self.truth_q, minlength=n_queries)
        self.counts = np.zeros(n_queries, dtype=np.int64)
        self.best_rank = np.full(len(keys), NOT_FOUND, dtype=np.int64)
        self.seconds = 0.0

    def add(self, q, t, rank):
        """Record one chunk of candidates (``q`` = query index, ``t`` = target, ``rank``)."""
        q = np.asarray(q, dtype=np.int64)
        self.counts += np.bincount(q, minlength=self.n_queries)
        if not len(self.truth_keys) or not len(q):
            return
        keys = (q << 32) | np.asarray(t, dtype=np.int64)
        idx = np.searchsorted(self.truth_keys, keys)
        idx[idx == len(self.truth_keys)] = 0
        hit = self.truth_keys[idx] == keys
        np.minimum.at(self.best_rank, idx[hit], np.asarray(rank, dtype=np.int64)[hit])

    def retrieved(self, k=None):
        """Boolean per true pair (in the original pair order): retrieved within top-``k``."""
        found = self.best_rank < (NOT_FOUND if k is None else k)
        out = np.empty_like(found)
        out[self.truth_order] = found
        return out

    def per_query(self, k=None):
        """(candidates, hits, n_true, oracle F0.5) per query."""
        found = self.best_rank < (NOT_FOUND if k is None else k)
        hits = np.bincount(self.truth_q[found], minlength=self.n_queries)
        counts = self.counts if k is None else np.minimum(self.counts, k)
        recall = np.divide(hits, self.n_true, out=np.zeros(self.n_queries), where=self.n_true > 0)
        f = np.where(self.n_true == 0, 1.0, f05_from_recall(recall))
        return counts, hits, self.n_true, f

    def summary(self, k=None, query_strata=None, pair_strata=None):
        """Overall metrics plus recall/oracle F0.5 per query stratum and per pair stratum.

        ``query_strata``: {name: bool mask over queries}; ``pair_strata``: {name: bool mask
        over true pairs in their original order}.
        """
        counts, hits, n_true, f = self.per_query(k)
        has = n_true > 0
        found_pairs = self.retrieved(k)
        out = {
            "n_queries": int(self.n_queries),
            "n_true_pairs": int(n_true.sum()),
            "candidates": _quantiles(counts),
            "pct_zero_candidates": round(100 * float((counts == 0).mean()), 3),
            "pair_recall": round(float(found_pairs.mean()) if len(found_pairs) else 0.0, 4),
            "s1_recall": round(float((hits[has] / n_true[has]).mean()) if has.any() else 0.0, 4),
            "pct_all_true_retrieved": round(100 * float((hits[has] == n_true[has]).mean()), 2),
            "pct_any_true_retrieved": round(100 * float((hits[has] > 0).mean()), 2),
            "oracle_f05": round(float(f.mean()), 4),
        }
        if query_strata:
            out["by_query_stratum"] = {
                name: {
                    "n_queries": int(m.sum()),
                    "mean_candidates": round(float(counts[m].mean()), 1) if m.any() else None,
                    "s1_recall": round(float((hits[m & has] / n_true[m & has]).mean()), 4) if (m & has).any() else None,
                    "oracle_f05": round(float(f[m].mean()), 4) if m.any() else None,
                }
                for name, m in query_strata.items()
            }
        if pair_strata:
            out["by_pair_stratum"] = {
                name: {"n_pairs": int(m.sum()),
                       "pair_recall": round(float(found_pairs[m].mean()), 4) if m.any() else None}
                for name, m in pair_strata.items()
            }
        return out
