"""E04: train the one stage-2 LightGBM matcher on the locked stage-1 candidates.

Stage 1 (locked): G+Indic retrieval, ``model_GI_unweighted.txt`` with its first 400 trees,
union-rank cap 500, top 50 per S1 (``ber.matcher.STAGE1``).

* Training S1: ``--train`` S1 from folds 1-4, excluding the 100k S1 the stage-1 ranker was
  trained on (so stage-1 scores are out-of-sample), stratified by country x match bucket.
  10% of them (hash of the S1 ID) are the early-stopping holdout.
* Dev S1: ``--valid`` S1 of the fold-0 dev subset (stratified), used only to pick the
  decision threshold and to report macro F0.5.
* Rows: every top-50 candidate; label 1 iff the pair is in the ground truth. No sampling.

Outputs: ``work/experiments/matcher/<out-name>/`` (never overwritten). Run from
code/business_entity_resolution/src:

    python ../experiments/e04_matcher.py --train 500 --valid 200 --out-name smoke --verify
    python ../experiments/e04_matcher.py --train 60000 --valid 20000 --out-name full
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e02_retrieval import peak_mb  # noqa: E402
from e03_ranker import POOLS, Context  # noqa: E402
from ber import matcher, ranker, tokens  # noqa: E402
from ber.config import Config  # noqa: E402
from ber.cv import stable_hash, stratified_subset  # noqa: E402

OUT_FILES = ("stage2_model.txt", "features.json", "config.json", "dev_metrics.json", "run_meta.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select_s1(split, n_train, n_valid):
    stage1_train = set(ranker.select_training_s1(split, 100_000))
    pool = split[(split["fold"] != 0) & ~split["entity_id"].isin(stage1_train)]
    train = pool["entity_id"].to_numpy()[stratified_subset(pool["entity_id"].to_numpy(), pool["stratum"].to_numpy(),
                                                             n_train, seed="ber-matcher-train")]
    dev = split[split["dev"]]
    valid = dev["entity_id"].to_numpy()[stratified_subset(dev["entity_id"].to_numpy(), dev["stratum"].to_numpy(),
                                                            n_valid, seed="ber-matcher-dev")]
    es = (stable_hash(list(train), "ber-matcher-es") % np.uint64(10_000)) < np.uint64(1_000)
    assert not stage1_train & set(train) and not set(train) & set(valid)
    return train, es, valid


def load_strings(cfg, ctx):
    """Query strings aligned with ``ctx.q_ids`` and target strings aligned with the pool."""
    s1 = tokens.read_normalized(cfg, "train", (1,), ["entity_id"] + matcher.STRING_COLUMNS)
    idx = pd.Index(s1["entity_id"].to_pandas().astype(str)).get_indexer(ctx.q_ids)
    assert (idx >= 0).all()
    s1 = s1.take(pa.array(idx))
    tt = tokens.read_normalized(cfg, "train", (2, 3), ["entity_id"] + matcher.STRING_COLUMNS)
    assert pc.all(pc.equal(tt["entity_id"], ctx.pool.entity_id)).as_py(), "target order differs from the pool"
    return ({c: s1[c].combine_chunks() for c in matcher.STRING_COLUMNS},
            {c: tt[c].combine_chunks() for c in matcher.STRING_COLUMNS})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=60_000)
    ap.add_argument("--valid", type=int, default=20_000)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--verify", action="store_true", help="extra consistency checks (smoke tests)")
    args = ap.parse_args(argv)
    t_start = time.time()
    log = lambda m: print(f"[{time.time() - t_start:7.1f}s | peak {peak_mb():6,.0f} MB] {m}", flush=True)
    cfg = Config.discover()
    out_dir = cfg.work_dir / "experiments" / "matcher" / args.out_name
    if any((out_dir / f).exists() for f in OUT_FILES):
        raise SystemExit(f"refusing to overwrite existing results in {out_dir}")
    import lightgbm as lgb

    st1 = matcher.STAGE1
    split = pq.read_table(cfg.work_dir / "cache" / "cv_v1" / "train_s1_split.parquet").to_pandas()
    train_ids, es_mask, valid_ids = select_s1(split, args.train, args.valid)
    log(f"S1: train {len(train_ids):,} (early-stopping holdout {int(es_mask.sum()):,}), dev {len(valid_ids):,}")
    model_path = cfg.work_dir / "experiments" / "ranker" / st1["model_file"]
    stage1 = lgb.Booster(model_file=str(model_path))
    stage1_names = ranker.feature_names(POOLS[st1["pool"]])
    assert stage1.feature_name() == stage1_names and stage1.num_trees() >= st1["trees"]
    names = matcher.stage2_feature_names(stage1_names)
    log(f"stage 1: {st1['model_file']} ({stage1.num_trees()} trees, using {st1['trees']}), cap {st1['cap']}, "
        f"top {st1['top_k']}; stage 2: {len(names)} features")

    ctx = Context(cfg, np.concatenate([train_ids, valid_ids]), log)
    qstr_all, tstr_all = load_strings(cfg, ctx)
    log("string columns loaded")
    index = pd.Index(ctx.q_ids)
    rows = index.get_indexer(np.concatenate([train_ids, valid_ids]))
    assert (rows >= 0).all()
    role = np.full(len(ctx.q_ids), -1, dtype=np.int8)  # 0 train, 1 early-stopping, 2 dev
    role[index.get_indexer(train_ids)] = np.where(es_mask, 1, 0)
    role[index.get_indexer(valid_ids)] = 2
    if args.verify:
        truth_sets = {}
        for q_, t_ in zip(ctx.truth_q.tolist(), ctx.truth_t.tolist()):
            truth_sets.setdefault(q_, set()).add(t_)

    parts = {k: [] for k in ("X", "y", "q", "t")}
    timing = dict.fromkeys(("retrieval_s", "stage1_features_s", "stage1_predict_s", "stage2_features_s"), 0.0)
    counts = {"retrieved": 0, "within_cap": 0, "top_k": 0}
    done, t_pass = 0, time.time()
    for block_rows, merged, qstats, dt in ctx.blocks(rows, args.chunk):
        timing["retrieval_s"] += dt
        m_full = merged[st1["pool"]]
        m, _ = matcher.cap_merged(m_full, st1["cap"])
        t0 = time.time()
        X1 = ctx.features(st1["pool"], m, qstats, block_rows)
        timing["stage1_features_s"] += time.time() - t0
        if args.verify:  # features on the capped rows == features on all rows restricted to the cap
            full = ctx.features(st1["pool"], m_full, qstats, block_rows)[m_full.union_rank < st1["cap"]]
            assert np.array_equal(full, X1, equal_nan=True), "capped features differ"
        t0 = time.time()
        top, raw, rank = matcher.stage1_top(m.q, m.t, X1, stage1, len(block_rows), threads=args.threads)
        timing["stage1_predict_s"] += time.time() - t0
        q_blk, t_glob = m.q[top], m.t[top]
        q_glob = block_rows[q_blk]
        t0 = time.time()
        qstr = {c: a.take(pa.array(q_glob)) for c, a in qstr_all.items()}
        tstr = {c: a.take(pa.array(t_glob)) for c, a in tstr_all.items()}
        X = matcher.stage2_features(X1[top], stage1_names, raw, rank, q_blk, len(block_rows), qstr, tstr)
        timing["stage2_features_s"] += time.time() - t0
        y = ranker.labels_for(q_glob, t_glob, ctx.truth_keys)
        if args.verify:
            ref = np.array([t_ in truth_sets.get(q_, ()) for q_, t_ in zip(q_glob.tolist(), t_glob.tolist())])
            assert np.array_equal(ref, y.astype(bool)), "labels differ from the ground-truth sets"
            assert (np.bincount(q_blk, minlength=len(block_rows)) <= st1["top_k"]).all()
            assert len(np.unique(ranker.pair_keys(q_glob, t_glob))) == len(q_glob), "duplicate pairs"
        counts["retrieved"] += len(m_full.q)
        counts["within_cap"] += len(m.q)
        counts["top_k"] += len(top)
        for k, v in (("X", X), ("y", y), ("q", q_glob), ("t", t_glob)):
            parts[k].append(v)
        done += len(block_rows)
        el = time.time() - t_pass
        log(f"progress {done:,}/{len(rows):,} S1 ({100 * done / len(rows):.1f}%), elapsed {el:.0f}s, "
            f"ETA {el / done * (len(rows) - done):.0f}s | " + " ".join(f"{k} {v:.0f}s" for k, v in timing.items()))
    data = {k: np.concatenate(v) for k, v in parts.items()}
    del parts
    timing = {k: round(v, 1) for k, v in timing.items()}
    timing["candidate_pass_s"] = round(time.time() - t_pass, 1)

    # ---- dataset summary
    r = role[data["q"]]
    X, y = data["X"], data["y"]
    summary = {}
    for label, sel in (("train", r == 0), ("early_stopping", r == 1), ("dev", r == 2)):
        summary[label] = {"rows": int(sel.sum()), "positives": int(y[sel].sum()),
                          "negatives": int((y[sel] == 0).sum()),
                          "positive_share": round(float(y[sel].mean()), 4) if sel.any() else None,
                          "s1": int(len(np.unique(data["q"][sel])))}
    true_in = {label: int(np.isin(ctx.truth_q, np.flatnonzero(role == code)).sum())
               for label, code in (("train", 0), ("early_stopping", 1), ("dev", 2))}
    for label in summary:
        summary[label]["true_pairs_of_these_s1"] = true_in[label]
        summary[label]["top_k_pair_recall"] = round(summary[label]["positives"] / max(true_in[label], 1), 4)
    nan_share = {n: round(float(v), 4) for n, v in zip(names, np.isnan(X).mean(axis=0))}
    n_inf = int(np.isinf(X).sum())
    log(f"rows: {json.dumps(summary)}")
    log(f"features {X.shape[1]}, inf values {n_inf}, columns with NaN "
        f"{ {n: v for n, v in nan_share.items() if v > 0} }")
    assert X.shape[1] == len(names) and n_inf == 0

    # ---- train
    t0 = time.time()
    fit, es = r == 0, r == 1
    ones = np.ones(len(y), np.float32)
    booster, evals = ranker.train_lgbm(X[fit], y[fit], ones[fit], X[es], y[es], ones[es], names)
    train_s = round(time.time() - t0, 1)
    best_it = booster.best_iteration
    es_auc = round(float(evals["val"]["auc"][best_it - 1]), 5)
    es_logloss = round(float(evals["val"]["binary_logloss"][best_it - 1]), 5)
    log(f"trained: {best_it} trees, early-stopping AUC {es_auc}, logloss {es_logloss}, {train_s}s")

    # ---- dev: threshold + one-S1-per-target, macro F0.5 over all dev S1
    dev = r == 2
    dev_rows = index.get_indexer(valid_ids)
    local = np.full(len(ctx.q_ids), -1, dtype=np.int64)
    local[dev_rows] = np.arange(len(dev_rows))
    qd, td, yd = local[data["q"][dev]], data["t"][dev], y[dev]
    pd_ = booster.predict(X[dev], num_threads=args.threads)
    n_true = np.bincount(local[ctx.truth_q[np.isin(ctx.truth_q, dev_rows)]], minlength=len(dev_rows))
    thr, best, curve = matcher.tune_threshold(qd, td, pd_, yd, len(dev_rows), n_true)
    no_1to1 = matcher.score_decision(qd, yd, pd_ >= thr, len(dev_rows), n_true)
    oracle = matcher.score_decision(qd, yd, yd == 1, len(dev_rows), n_true)
    log(f"dev: threshold {thr} -> {best}; same threshold without one-S1-per-target: {no_1to1['macro_f05']}; "
        f"oracle on top-{st1['top_k']}: {oracle['macro_f05']}")
    selected = matcher.decide(qd, td, pd_, thr)
    assert len(np.unique(td[selected])) == int(selected.sum()), "a target is assigned to more than one S1"
    checks = {"one_s1_per_target": True}
    if args.verify:  # cross-check the metric against the official-rule scorer in src/evaluation.py
        import evaluation

        dev_ids_local = np.asarray(ctx.q_ids)[dev_rows]
        tid = ctx.pool.entity_id
        truth = {s: set() for s in dev_ids_local}
        for q_, t_ in zip(local[ctx.truth_q[np.isin(ctx.truth_q, dev_rows)]].tolist(),
                          ctx.truth_t[np.isin(ctx.truth_q, dev_rows)].tolist()):
            truth[dev_ids_local[q_]].add(tid[t_].as_py())
        pred = {s: set() for s in dev_ids_local}
        for q_, t_ in zip(qd[selected].tolist(), td[selected].tolist()):
            pred[dev_ids_local[q_]].add(tid[t_].as_py())
        ref = evaluation.evaluate(pred, truth)["macro_f0.5"]
        assert abs(ref - best["macro_f05"]) < 1e-5, (ref, best["macro_f05"])
        checks.update(evaluation_py_macro_f05=round(ref, 5), capped_features_equal=True, labels_equal=True)
        log(f"verify: evaluation.py macro F0.5 {ref:.5f} == matcher {best['macro_f05']:.5f}")

    # ---- save
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "stage2_model.txt"))
    (out_dir / "features.json").write_text(json.dumps(names, indent=1), encoding="utf-8")
    config = {"stage1": st1, "stage1_model_sha256": sha256(model_path), "stage2_features": names,
              "lgb_params": ranker.LGB_PARAMS, "rounds": 2000, "early_stop": 50, "threshold": thr,
              "one_s1_per_target": True,
              "selection": {"train": f"{args.train} stratified S1 of folds 1-4 minus the 100k stage-1 training S1 "
                                     "(seed ber-matcher-train); 10% early-stopping holdout (hash seed ber-matcher-es)",
                            "dev": f"{args.valid} stratified S1 of the fold-0 dev subset (seed ber-matcher-dev)"}}
    (out_dir / "config.json").write_text(json.dumps(config, indent=1, default=str), encoding="utf-8")
    metrics = {"threshold": thr, "dev": best, "dev_same_threshold_without_one_to_one": no_1to1,
               "dev_oracle_top_k": oracle, "threshold_curve": curve,
               "early_stopping": {"best_iteration": best_it, "auc": es_auc, "logloss": es_logloss}}
    (out_dir / "dev_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    gain = booster.feature_importance("gain")
    meta = {"args": vars(args), "rows": summary, "candidates": counts, "timing": timing, "train_s": train_s,
            "index_build_s": round(ctx.index_seconds, 1), "nan_share": nan_share, "checks": checks,
            "top_features_by_gain": [[names[i], round(float(gain[i] / gain.sum()), 4)] for i in np.argsort(-gain)[:20]],
            "stage2_model_sha256": sha256(out_dir / "stage2_model.txt"),
            "stage1_model_sha256": config["stage1_model_sha256"],
            "peak_working_set_mb": round(peak_mb()), "total_s": round(time.time() - t_start, 1)}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    log(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
