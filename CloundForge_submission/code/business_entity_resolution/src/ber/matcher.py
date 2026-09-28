"""Stage-2 matcher: pair features on the stage-1 top-K, one LightGBM model, decision rule.

Pipeline per block of queries (all vectorized, no per-pair Python loops):

1. ``stage1_top``: keep the candidates with union IDF rank < ``cap`` (a per-query prefix of
   ``Merged``), score them with the first ``trees`` trees of the stage-1 ranker (raw margin)
   and keep the top ``top_k`` per query. These rows are the stage-2 candidate set.
2. ``stage2_features``: stage-1 features (without ``country``, which is not usable for the
   unseen test country), stage-1 ranking context and RapidFuzz string similarities
   (``process.cpdist``, all cores).
3. A LightGBM binary classifier gives p(match).
4. ``decide``: keep a pair when it is the best-scoring S1 of its target (each S2/S3 record
   matches at most one S1 in the ground truth) and p >= threshold.

``macro_f05`` follows the official definition (``src/evaluation.py``: per-S1 F0.5, 1.0 for
an empty prediction on a singleton, averaged over every S1, singletons included).
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import distance, fuzz, process

from . import ranker

STAGE1 = {"pool": "GI", "model_file": "model_GI_unweighted.txt", "trees": 400, "cap": 500, "top_k": 50}
STRING_COLUMNS = ["name_fold", "name_stripped", "address_norm", "legal_forms", "address_numbers"]
DROP_STAGE1 = ("country",)
CONTEXT_NAMES = ["s1_score", "s1_rank", "s1_gap", "s1_best", "s1_gap12", "s1_ntop"]
STRING_FEATURES = [  # (name, column, scorer)
    ("name_ratio", "name_fold", fuzz.ratio),
    ("name_token_set", "name_fold", fuzz.token_set_ratio),
    ("name_token_sort", "name_fold", fuzz.token_sort_ratio),
    ("name_partial", "name_fold", fuzz.partial_ratio),
    ("name_jaro_winkler", "name_fold", distance.JaroWinkler.normalized_similarity),
    ("stripped_ratio", "name_stripped", fuzz.ratio),
    ("stripped_token_set", "name_stripped", fuzz.token_set_ratio),
    ("addr_ratio", "address_norm", fuzz.ratio),
    ("addr_token_set", "address_norm", fuzz.token_set_ratio),
    ("addr_partial", "address_norm", fuzz.partial_ratio),
    ("addr_numbers_token_set", "address_numbers", fuzz.token_set_ratio),
]
EXTRA_NAMES = ["legal_form_equal", "stripped_first_token_equal"]


def stage2_feature_names(stage1_names):
    keep = [n for n in stage1_names if n not in DROP_STAGE1]
    return keep + CONTEXT_NAMES + [n for n, _, _ in STRING_FEATURES] + EXTRA_NAMES


# --------------------------------------------------------------------------- stage-1 candidates

def cap_merged(m, cap):
    """``Merged`` restricted to union rank < ``cap``: a prefix of each query's rows, so the
    per-query offsets are recomputed and ``pair_features`` gives the same values as on ``m``."""
    keep = m.union_rank < cap
    counts = np.bincount(m.q[keep], minlength=len(m.starts) - 1)
    starts = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    return ranker.Merged(q=m.q[keep], t=m.t[keep], score=m.score[keep], rank=m.rank[keep], mask=m.mask[keep],
                         union_score=m.union_score[keep], union_rank=m.union_rank[keep],
                         n_cand=m.n_cand[keep], starts=starts), np.flatnonzero(keep)


def stage1_top(q, t, X1, booster, n_queries, trees=STAGE1["trees"], top_k=STAGE1["top_k"], threads=None):
    """Rows (into q/t/X1) of the top ``top_k`` per query by the first ``trees`` trees, with
    their raw stage-1 score and rank (0 = best)."""
    raw = booster.predict(X1, raw_score=True, num_iteration=trees, num_threads=threads or 0)
    rank = ranker.rank_by_score(q, t, raw, n_queries)
    rows = np.flatnonzero(rank < top_k)
    return rows, raw[rows].astype(np.float32), rank[rows].astype(np.float32)


# --------------------------------------------------------------------------- stage-2 features

def _strings(arr):
    arr = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    return pc.fill_null(arr, "")


def _first_token(arr):
    return pc.fill_null(pc.list_element(pc.split_pattern(arr, " ", max_splits=1), 0), "")


def stage2_features(X1, stage1_names, raw, rank, q, n_queries, qstr, tstr, workers=-1):
    """Float32 matrix with the columns of ``stage2_feature_names(stage1_names)``.

    ``X1``: stage-1 features of the rows; ``raw``/``rank``: stage-1 score and rank; ``q``:
    block-relative query of each row; ``qstr``/``tstr``: {column: pyarrow string array}
    aligned with the rows (query side and target side).
    """
    n = len(q)
    keep = [j for j, name in enumerate(stage1_names) if name not in DROP_STAGE1]
    best = np.full(n_queries, np.nan, dtype=np.float32)
    second = np.full(n_queries, np.nan, dtype=np.float32)
    best[q[rank == 0]] = raw[rank == 0]
    second[q[rank == 1]] = raw[rank == 1]
    ntop = np.bincount(q, minlength=n_queries).astype(np.float32)
    cols = [raw, rank, best[q] - raw, best[q], (best - second)[q], ntop[q]]

    strings = {c: (_strings(qstr[c]), _strings(tstr[c])) for c in STRING_COLUMNS}
    empty = {c: (pc.equal(a, "").to_numpy(zero_copy_only=False) | pc.equal(b, "").to_numpy(zero_copy_only=False))
             for c, (a, b) in strings.items()}
    as_list = {}
    for name, col, scorer in STRING_FEATURES:
        if col not in as_list:
            as_list[col] = tuple(s.to_numpy(zero_copy_only=False) for s in strings[col])
        v = process.cpdist(*as_list[col], scorer=scorer, workers=workers, dtype=np.float32)
        if scorer is distance.JaroWinkler.normalized_similarity:
            v = v * 100
        v[empty[col]] = np.nan
        cols.append(v)
    la, lb = strings["legal_forms"]
    legal = pc.equal(la, lb).to_numpy(zero_copy_only=False).astype(np.float32)
    legal[empty["legal_forms"]] = np.nan
    fa, fb = _first_token(strings["name_stripped"][0]), _first_token(strings["name_stripped"][1])
    first = pc.equal(fa, fb).to_numpy(zero_copy_only=False).astype(np.float32)
    first[empty["name_stripped"]] = np.nan
    cols += [legal, first]

    X = np.empty((n, len(keep) + len(cols)), dtype=np.float32)
    X[:, :len(keep)] = X1[:, keep]
    for j, col in enumerate(cols):
        X[:, len(keep) + j] = col
    X[np.isinf(X)] = np.nan
    return X


# --------------------------------------------------------------------------- decision and metric

def best_s1_per_target(q, t, p):
    """True for the highest-p pair of each target (ties: smaller query index)."""
    order = np.lexsort((q, -p, t))
    ts = t[order]
    first = np.r_[True, ts[1:] != ts[:-1]]
    out = np.zeros(len(t), dtype=bool)
    out[order[first]] = True
    return out


def decide(q, t, p, threshold, one_to_one=True):
    """Boolean over pairs: predicted matches. With ``one_to_one`` a target goes to its
    best-scoring S1 only (equivalent to thresholding first, since the best pair of a
    target is the last one to fall below any threshold)."""
    sel = p >= threshold
    return sel & best_s1_per_target(q, t, p) if one_to_one else sel


def macro_f05(n_pred, n_true, n_hit, beta=0.5):
    """Mean per-S1 F-beta with the official rules (arrays over all evaluated S1)."""
    n_pred, n_true, n_hit = (np.asarray(a, dtype=np.float64) for a in (n_pred, n_true, n_hit))
    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = n_hit / n_pred
        rec = n_hit / n_true
        f = (1 + b2) * prec * rec / (b2 * prec + rec)
    f = np.where((n_pred == 0) | (n_hit == 0), 0.0, f)
    f = np.where(n_true == 0, (n_pred == 0).astype(np.float64), f)
    return float(f.mean())


def score_decision(q, y, selected, n_queries, n_true):
    """Macro F0.5 and totals for a boolean selection of pairs (``q`` in [0, n_queries))."""
    n_pred = np.bincount(q[selected], minlength=n_queries)
    n_hit = np.bincount(q[selected], weights=y[selected], minlength=n_queries)
    return {"macro_f05": round(macro_f05(n_pred, n_true, n_hit), 5), "pred_pairs": int(n_pred.sum()),
            "true_positive_pairs": int(n_hit.sum()),
            "pair_precision": round(float(n_hit.sum() / max(n_pred.sum(), 1)), 4),
            "pair_recall": round(float(n_hit.sum() / max(n_true.sum(), 1)), 4),
            "pct_s1_predicted_empty": round(100 * float((n_pred == 0).mean()), 2)}


def tune_threshold(q, t, p, y, n_queries, n_true, grid=None, one_to_one=True):
    """Best threshold on ``grid`` by macro F0.5; returns (threshold, metrics, curve)."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    best_mask = best_s1_per_target(q, t, p) if one_to_one else np.ones(len(p), bool)
    curve = {}
    for thr in grid:
        curve[float(thr)] = score_decision(q, y, (p >= thr) & best_mask, n_queries, n_true)
    thr = max(curve, key=lambda k: (curve[k]["macro_f05"], -k))
    return thr, curve[thr], curve
