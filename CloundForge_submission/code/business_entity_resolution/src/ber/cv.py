"""Deterministic S1-level cross-validation folds and the development subset.

The unit of splitting is the S1 entity (S1 is deduplicated), so all candidate pairs of an
S1 stay in one fold. Assignment is stratified by ``country × match bucket`` where the
bucket is the number of true matches: ``0``, ``1``, ``2-3`` or ``4+``.

Determinism: every S1 gets a keyed BLAKE2b hash of its entity ID. Within each stratum the
S1s are ordered by that hash (ID as tie-break) and folds are dealt round-robin, so the
result depends only on the IDs, their strata and the seed, never on file order.

The development subset is taken **inside fold 0** (so it is always held out together with
that fold): per stratum, the S1s with the smallest hashes, in numbers proportional to the
stratum's share of fold 0 (largest-remainder rounding so the total is exact).
"""

import hashlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import io

N_FOLDS = 5
DEV_SIZE = 100_000
SEED = "ber-cv-v1"
BUCKETS = ("0", "1", "2-3", "4+")


def match_bucket(n_matches):
    """Bucket labels for an array of match counts."""
    n = np.asarray(n_matches)
    return np.select([n == 0, n == 1, n <= 3], ["0", "1", "2-3"], "4+")


def stable_hash(ids, seed=SEED):
    """64-bit keyed BLAKE2b hash of each string ID (uint64 array)."""
    key = seed.encode()
    return np.fromiter(
        (int.from_bytes(hashlib.blake2b(i.encode(), digest_size=8, key=key).digest(), "little") for i in ids),
        dtype=np.uint64, count=len(ids),
    )


def _order_within_strata(ids, strata, hashes):
    """Positions sorted by (stratum, hash, id) and the rank of each position in its stratum."""
    frame = pd.DataFrame({"stratum": np.asarray(strata, dtype=object), "hash": hashes,
                          "id": np.asarray(ids, dtype=object)})
    order = frame.sort_values(["stratum", "hash", "id"], kind="mergesort").index.to_numpy()
    sorted_strata = frame["stratum"].to_numpy()[order]
    new_group = np.r_[True, sorted_strata[1:] != sorted_strata[:-1]]
    group_start = np.maximum.accumulate(np.where(new_group, np.arange(len(order)), 0))
    return order, np.arange(len(order)) - group_start


def assign_folds(ids, strata, n_folds=N_FOLDS, seed=SEED):
    """Fold number (int8) per ID, dealt round-robin by hash within each stratum."""
    order, rank = _order_within_strata(ids, strata, stable_hash(ids, seed))
    folds = np.empty(len(order), dtype=np.int8)
    folds[order] = (rank % n_folds).astype(np.int8)
    return folds


def stratified_subset(ids, strata, n, seed=SEED):
    """Boolean mask selecting ``n`` IDs, proportionally per stratum, smallest hashes first."""
    strata = np.asarray(strata, dtype=object)
    labels, sizes = np.unique(strata, return_counts=True)
    n = min(n, len(ids))
    exact = sizes * n / len(ids)
    quota = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - quota), kind="stable")[: n - quota.sum()]:
        quota[i] += 1
    order, rank = _order_within_strata(ids, strata, stable_hash(ids, seed + "-subset"))
    limit = dict(zip(labels, quota))
    chosen = np.zeros(len(ids), dtype=bool)
    chosen[order] = rank < np.array([limit[s] for s in strata[order]])
    return chosen


def build_train_split(cfg, n_folds=N_FOLDS, dev_size=DEV_SIZE, seed=SEED):
    """DataFrame of every train S1: entity_id, country, n_matches, bucket, stratum, fold, dev."""
    s1 = io.read_source_table(cfg, "train", 1)
    gt = io.load_ground_truth(cfg).s1
    df = pd.DataFrame({"entity_id": s1["entity_id"].to_pylist(), "country": s1["country"].to_pylist()})
    n_matches = pd.Series(gt["n_matches"].to_numpy(), index=gt["source1_entity_id"].astype(str))
    df["n_matches"] = n_matches.reindex(df["entity_id"]).to_numpy()
    if df["n_matches"].isna().any():
        raise ValueError("train S1 without a ground-truth row")
    df["n_matches"] = df["n_matches"].astype(np.int16)
    df["bucket"] = match_bucket(df["n_matches"].to_numpy())
    df["stratum"] = df["country"] + "|" + df["bucket"]
    df["fold"] = assign_folds(df["entity_id"].to_numpy(), df["stratum"].to_numpy(), n_folds, seed)
    df["dev"] = False
    in_fold0 = np.flatnonzero(df["fold"].to_numpy() == 0)
    dev = stratified_subset(df["entity_id"].to_numpy()[in_fold0], df["stratum"].to_numpy()[in_fold0], dev_size, seed)
    df.loc[in_fold0[dev], "dev"] = True
    return df


def save_split(cfg, df, name="train_s1_split"):
    """Write the split to ``work/cache/cv_v1/<name>.parquet`` and return the path."""
    path = cfg.work_dir / "cache" / "cv_v1" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    return path
