"""Stage-1 ranker: cheap pair features on retrieved candidates and a LightGBM scorer.

The ranker only re-orders candidates that retrieval already produced (never the whole
target pool) so that a small top-K per S1 keeps as many true pairs as possible. It is
trained with the S1-level CV split (folds 1-4) and evaluated on the held-out dev subset.

Pipeline per chunk of queries:

1. ``merge_channels``: channel outputs -> one row per unique (query, target) with per-channel
   score and rank (NaN when the channel did not retrieve the pair), union score/rank, mask.
2. ``pair_features``: a float32 matrix with the columns of ``feature_names`` (vectorized,
   no string work per pair). Sources: the merged scores, per-query block statistics
   (``query_block_stats``) and per-record arrays (``RecordArrays``) built once per pool.
3. ``LightGBM`` binary classifier (label 1 = true S1-target pair); its probability is the
   ranking score. ``rank_by_score`` orders each query's candidates by that score.

No feature uses labels, other queries' candidates or target ownership.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from .retrieval.sparse_topk import rank_rows

SCRIPT_CODES = {"empty": 0, "latin": 1, "mixed": 2, "other": 3}  # every Indic script -> 4
COUNTRY_CODES = {"US": 0, "India": 1}  # any other country -> 2 (open set)


def script_code(labels):
    labels = np.asarray(labels, dtype=object)
    out = np.full(len(labels), 4, dtype=np.int8)
    for label, code in SCRIPT_CODES.items():
        out[labels == label] = code
    return out


def country_code(labels):
    labels = np.asarray(labels, dtype=object)
    out = np.full(len(labels), 2, dtype=np.int8)
    for label, code in COUNTRY_CODES.items():
        out[labels == label] = code
    return out


def first_value_hash(values):
    """uint64 hash of the first space-separated value of each string; 0 when empty."""
    arr = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    first = pc.list_element(pc.split_pattern(arr, " ", max_splits=1), 0)
    first = pc.fill_null(first, "").to_numpy(zero_copy_only=False)
    h = pd.util.hash_array(first.astype(object), categorize=True).astype(np.uint64)
    h[first == ""] = 0
    return h


# --------------------------------------------------------------------------- per-record arrays

@dataclass
class RecordArrays:
    """Per-record (query or target) arrays in global order, used to build pair features."""

    country: np.ndarray
    script: np.ndarray
    address_missing: np.ndarray
    house: np.ndarray
    unit: np.ndarray
    postcode: np.ndarray
    source: np.ndarray = None
    extra: dict = field(default_factory=dict)  # name_ntok, addr_ntok, key_df, mass_<channel>

    @classmethod
    def from_table(cls, table):
        """From a normalized table with country, name_script, address_missing, number columns."""
        return cls(
            country=country_code(table["country"].to_numpy(zero_copy_only=False)),
            script=script_code(table["name_script"].to_numpy(zero_copy_only=False)),
            address_missing=table["address_missing"].to_numpy(zero_copy_only=False).astype(np.int8),
            house=first_value_hash(table["house_numbers"]),
            unit=first_value_hash(table["unit_numbers"]),
            postcode=first_value_hash(table["postcode"]),
            source=table["source"].to_numpy().astype(np.int8) if "source" in table.column_names else None,
        )


def add_target_statistics(arrays, pool, channels, extra_pools=None):
    """Fill ``arrays.extra`` with per-target token counts, key DF and per-channel IDF mass.

    ``channels``: {feature prefix: Channel}. The mass of a channel is the summed IDF of the
    target's tokens that the channel does not suppress (DF <= cap), i.e. the most a query
    could share with it. ``extra_pools``: {prefix: {country: CountryPool}} for channels that
    index a subset of the targets (e.g. the Indic skeleton key).
    """
    n = len(arrays.country)
    for name in ("name_ntok", "addr_ntok", "key_df"):
        arrays.extra[name] = np.zeros(n, dtype=np.float32)
    for prefix in channels:
        arrays.extra[f"tmass_{prefix}"] = np.zeros(n, dtype=np.float32)
    for c, cp in pool.countries.items():
        for field_name, out in (("name_fold", "name_ntok"), ("address_norm", "addr_ntok")):
            inv = cp.fields[field_name].inverted
            arrays.extra[out][cp.target_idx] = np.bincount(inv.indices, minlength=inv.shape[1])
        key = cp.fields["name_key"]
        arrays.extra["key_df"][cp.target_idx[key.inverted.indices]] = np.repeat(key.df, np.diff(key.inverted.indptr))
    for prefix, ch in channels.items():
        pools = (extra_pools or {}).get(prefix) or pool.countries
        for c, cp in pools.items():
            fi = cp.fields[ch.field]
            w = fi.idf if ch.df_cap is None else np.where(fi.df <= ch.df_cap, fi.idf, 0).astype(np.float32)
            arrays.extra[f"tmass_{prefix}"][cp.target_idx] = fi.inverted.T @ w
    return arrays


def query_block_stats(main, runners, channels, start, stop):
    """Per-query statistics of one block: capped IDF mass per channel, name/address token
    counts, the largest name-token IDF and the DF of the query's strong-name key.

    ``main``: the runner holding the ``name_fold``/``address_norm``/``name_key`` queries;
    ``runners``: {prefix: ChannelRunner} (a channel whose runner is missing gets mass 0).
    """
    n = stop - start
    out = {}
    for prefix, ch in channels.items():
        runner = runners.get(prefix)
        out[f"qmass_{prefix}"] = (np.asarray(runner.capped(ch.field, ch.df_cap)[start:stop].sum(axis=1)).ravel()
                                  .astype(np.float32) if runner else np.zeros(n, np.float32))
    name = main.queries["name_fold"][start:stop]
    out["q_name_ntok"] = np.diff(name.indptr).astype(np.float32)
    out["q_addr_ntok"] = np.diff(main.queries["address_norm"][start:stop].indptr).astype(np.float32)
    out["q_name_maxidf"] = np.asarray(name.max(axis=1).todense()).ravel().astype(np.float32)
    key = main.queries["name_key"][start:stop]
    kdf = np.zeros(n, dtype=np.float32)
    has = np.diff(key.indptr) > 0
    kdf[has] = main.pool.fields["name_key"].df[key.indices]
    out["q_key_df"] = kdf
    return out


# --------------------------------------------------------------------------- merging and features

@dataclass
class Merged:
    """Unique (query, target) rows of one block, sorted by query then union rank."""

    q: np.ndarray            # block-relative query row (int32)
    t: np.ndarray            # global target index (int32)
    score: np.ndarray        # (n, C) float32 per-channel score, NaN if absent
    rank: np.ndarray         # (n, C) float32 per-channel rank, NaN if absent
    mask: np.ndarray         # uint8 channel bits
    union_score: np.ndarray  # summed channel scores
    union_rank: np.ndarray   # rank within the query by union score (0 = best)
    n_cand: np.ndarray       # candidates of the row's query (per row)
    starts: np.ndarray       # start offset of each block query in the sorted rows


def merge_channels(parts, n_queries):
    """Merge [(q, t, score, rank), ...] (one per channel, fixed order) into ``Merged``."""
    n_ch = len(parts)
    q_all = np.concatenate([p[0] for p in parts]).astype(np.int64)
    t_all = np.concatenate([p[1] for p in parts]).astype(np.int64)
    keys, inverse = np.unique((q_all << 32) | t_all, return_inverse=True)
    n = len(keys)
    score = np.full((n, n_ch), np.nan, dtype=np.float32)
    rank = np.full((n, n_ch), np.nan, dtype=np.float32)
    mask = np.zeros(n, dtype=np.uint8)
    offset = 0
    for c, (_, _, s, r) in enumerate(parts):
        idx = inverse[offset:offset + len(s)]
        score[idx, c] = s
        rank[idx, c] = r
        mask[idx] |= np.uint8(1 << c)
        offset += len(s)
    q = (keys >> 32).astype(np.int32)
    t = (keys & 0xFFFFFFFF).astype(np.int32)
    union_score = np.nansum(score, axis=1).astype(np.float32)
    order, urank = rank_rows(q, t, union_score, n_queries)
    q, t = q[order], t[order]
    counts = np.bincount(q, minlength=n_queries)
    starts = np.zeros(n_queries + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    return Merged(q=q, t=t, score=score[order], rank=rank[order], mask=mask[order],
                  union_score=union_score[order], union_rank=urank.astype(np.int32),
                  n_cand=counts[q].astype(np.float32), starts=starts)


def feature_names(prefixes):
    names = []
    for p in prefixes:
        names += [f"s_{p}", f"r_{p}", f"qfrac_{p}", f"tfrac_{p}", f"wjac_{p}"]
    names += ["s_sum", "r_sum", "s_rel", "log_n_cand", "n_channels",
              "house_eq", "house_conflict", "unit_eq", "unit_conflict", "post_eq", "post_conflict",
              "t_source", "country", "q_script", "t_script", "q_addr_missing", "t_addr_missing",
              "q_name_ntok", "t_name_ntok", "q_addr_ntok", "t_addr_ntok", "q_name_maxidf",
              "q_key_df", "t_key_df"]
    return names


def pair_features(m, prefixes, qstats, q_global, queries, targets):
    """Float32 feature matrix (rows of ``m``, columns ``feature_names(prefixes)``).

    ``qstats``: output of ``query_block_stats`` for the block; ``q_global``: global query
    index of each block row; ``queries``/``targets``: ``RecordArrays``.
    """
    n = len(m.q)
    cols = []
    qg = q_global[m.q]
    t = m.t
    with np.errstate(divide="ignore", invalid="ignore"):
        for c, p in enumerate(prefixes):
            s = m.score[:, c]
            qm = qstats[f"qmass_{p}"][m.q]
            tm = targets.extra[f"tmass_{p}"][t]
            cols += [s, m.rank[:, c], s / qm, s / tm, s / (qm + tm - s)]
        best = m.union_score[m.starts[m.q]]
        cols += [m.union_score, m.union_rank.astype(np.float32), m.union_score / best,
                 np.log1p(m.n_cand), np.unpackbits(m.mask[:, None], axis=1).sum(axis=1)]
    for a, b in ((queries.house[qg], targets.house[t]), (queries.unit[qg], targets.unit[t]),
                 (queries.postcode[qg], targets.postcode[t])):
        both = (a != 0) & (b != 0)
        cols += [both & (a == b), both & (a != b)]
    cols += [targets.source[t], queries.country[qg], queries.script[qg], targets.script[t],
             queries.address_missing[qg], targets.address_missing[t],
             qstats["q_name_ntok"][m.q], targets.extra["name_ntok"][t],
             qstats["q_addr_ntok"][m.q], targets.extra["addr_ntok"][t], qstats["q_name_maxidf"][m.q],
             qstats["q_key_df"][m.q], targets.extra["key_df"][t]]
    X = np.empty((n, len(cols)), dtype=np.float32)
    for j, col in enumerate(cols):
        X[:, j] = col
    return X


# --------------------------------------------------------------------------- labels, sampling, ranking

def pair_keys(q, t):
    return (np.asarray(q, dtype=np.int64) << 32) | np.asarray(t, dtype=np.int64)


def labels_for(q_global, t, truth_keys_sorted):
    """1 where (query, target) is a true pair (``truth_keys_sorted`` from ``pair_keys``)."""
    keys = pair_keys(q_global, t)
    idx = np.searchsorted(truth_keys_sorted, keys)
    idx[idx == len(truth_keys_sorted)] = 0
    return (truth_keys_sorted[idx] == keys).astype(np.int8) if len(truth_keys_sorted) else np.zeros(len(keys), np.int8)


def sample_negatives(labels, union_rank, rng, hard_rank=200, hard_rate=0.05, rest_rate=0.005):
    """Row selection for training: all positives, negatives sampled by union-rank region.

    Returns ``(selected_row_indices, weight, is_hard)``; ``weight`` is the inverse sampling
    rate (1 for positives), so a weighted model sees the real candidate distribution.
    """
    hard = union_rank < hard_rank
    u = rng.random(len(labels))
    keep_neg = (labels == 0) & np.where(hard, u < hard_rate, u < rest_rate)
    sel = np.flatnonzero((labels == 1) | keep_neg)
    weight = np.where(labels[sel] == 1, 1.0, np.where(hard[sel], 1 / hard_rate, 1 / rest_rate)).astype(np.float32)
    return sel, weight, hard[sel]


def rank_by_score(q, t, score, n_queries):
    """In-query rank (0 = best) by ``score`` descending, target index ascending on ties."""
    order, rank = rank_rows(np.asarray(q), np.asarray(t), np.asarray(score, dtype=np.float32), n_queries)
    out = np.empty(len(order), dtype=np.int32)
    out[order] = rank
    return out


LGB_PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
    "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "metric": ["binary_logloss", "auc"], "verbose": -1, "seed": 0, "deterministic": True,
    "force_col_wise": True, "num_threads": 8,
}


def train_lgbm(X, y, w, X_val, y_val, w_val, names, params=None, rounds=2000, early_stop=50):
    """Train a LightGBM binary classifier with early stopping on a held-out S1 group."""
    import lightgbm as lgb

    params = dict(LGB_PARAMS, **(params or {}))
    dtrain = lgb.Dataset(X, y, weight=w, feature_name=list(names), free_raw_data=False)
    dval = lgb.Dataset(X_val, y_val, weight=w_val, reference=dtrain)
    evals = {}
    booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dval], valid_names=["val"],
                        callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.record_evaluation(evals)])
    return booster, evals


# --------------------------------------------------------------------------- S1-level selection

def select_training_s1(split, n, seed="ber-ranker-v1"):
    """Stratified sample of ``n`` training S1 IDs from folds 1-4 only (never fold 0 / dev)."""
    from .cv import stratified_subset

    pool = split[split["fold"] != 0]
    chosen = stratified_subset(pool["entity_id"].to_numpy(), pool["stratum"].to_numpy(), n, seed)
    return pool["entity_id"].to_numpy()[chosen]


def holdout_mask(s1_ids, frac=0.1, seed="ber-ranker-holdout"):
    """True for S1 IDs in the early-stopping holdout (hash-based, so whole S1s move together)."""
    from .cv import stable_hash

    h = stable_hash(list(s1_ids), seed)
    return (h % np.uint64(10_000)) < np.uint64(int(frac * 10_000))
