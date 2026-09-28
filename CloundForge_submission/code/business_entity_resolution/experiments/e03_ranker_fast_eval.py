"""E03c: fast re-evaluation of the saved stage-1 LightGBM rankers (no training).

Loads the models written by ``e03_ranker.py`` (read-only) and scores them on the same
early-stopping holdout that script would have used for its configuration choice:

* Queries: the 10% hash holdout of the 100k stratified training S1 (folds 1-4, never
  fold 0 / dev). It inherits the country x match-bucket stratification of the training
  sample; ``--n`` takes a stratified subset of it (for smoke tests).
* Speed: the full holdout pass scored every retrieved candidate (~4k per query) with every
  model. Here only each query's top ``--cap`` candidates by union IDF rank are scored; the
  rest are ranked after them, so they can never enter a top-K with K <= 200 < cap. The
  recall this gives up is reported (``cap_ceiling``: true pairs with union rank >= cap).
* Models: G/unweighted, G/hard_only, G+I/unweighted, G+I/hard_only (the two weighted models
  stopped at 1 tree). The union-IDF rank of each pool is reported as a free baseline.
* Selection: per pool, the configuration with the best mean oracle F0.5 over K in
  {20, 50, 100, 200} (same rule as e03_ranker).

Outputs (never overwritten; the original ``ranker/`` JSON files are not touched):
``work/experiments/ranker_fast_eval/`` for the full holdout, ``.../smoke_n<N>/`` otherwise.
Run from code/business_entity_resolution/src:

    python ../experiments/e03_ranker_fast_eval.py [--n 100] [--cap 1000] [--chunk 500]
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e02_retrieval import peak_mb  # noqa: E402
from e03_ranker import POOLS, SELECT_K, Context, curve  # noqa: E402
from ber import ranker  # noqa: E402
from ber.config import Config  # noqa: E402
from ber.cv import match_bucket, stratified_subset  # noqa: E402
from ber.retrieval.evaluate import RetrievalEvaluator  # noqa: E402

MODELS = {"G/unweighted": ("G", "model_G_unweighted.txt"), "G/hard_only": ("G", "model_G_hard_only.txt"),
          "GI/unweighted": ("GI", "model_GI_unweighted.txt"), "GI/hard_only": ("GI", "model_GI_hard_only.txt")}
CAP_PROBES = (200, 300, 500, 1000, 2000, 5000)
RESULT_FILES = ("fast_eval_results.json", "fast_eval_summary.csv", "run_meta.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_models(model_dir, models, log):
    import lightgbm as lgb

    boosters, info = {}, {}
    for name, (pool, fname) in models.items():
        path = model_dir / fname
        booster = lgb.Booster(model_file=str(path))
        expected = ranker.feature_names(POOLS[pool])
        if booster.feature_name() != expected:
            raise ValueError(f"{fname}: feature names differ from ranker.feature_names({pool})")
        boosters[name] = booster
        info[name] = {"file": str(path), "sha256": sha256(path), "trees": booster.num_trees(),
                      "n_features": booster.num_feature()}
        log(f"loaded {fname}: {booster.num_trees()} trees, {booster.num_feature()} features")
    return boosters, info


def holdout_queries(split, n):
    """The e03_ranker holdout S1 IDs; a stratified subset of ``n`` of them when n is smaller."""
    train_ids = ranker.select_training_s1(split, 100_000)
    hold_ids = train_ids[ranker.holdout_mask(train_ids)]
    if n and n < len(hold_ids):
        strata = split.set_index("entity_id").loc[hold_ids, "stratum"].to_numpy()
        hold_ids = hold_ids[stratified_subset(hold_ids, strata, n, seed="ber-ranker-fast-eval")]
    return hold_ids, len(train_ids)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="stratified subset of the holdout (default: all ~10k)")
    ap.add_argument("--cap", type=int, default=1000, help="score only the top-CAP candidates by union rank")
    ap.add_argument("--chunk", type=int, default=500, help="queries per block (progress is logged per block)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--trees", type=int, nargs="+",
                    help="also score each model with only its first N trees (one pass: raw scores are summed "
                         "over consecutive tree ranges); the largest cutoff is the reference for top-K overlap")
    ap.add_argument("--overlap-k", type=int, default=50, help="top-K whose identity is compared across --trees")
    ap.add_argument("--out-name", help="output subfolder of ranker_fast_eval (default: derived from --n)")
    args = ap.parse_args(argv)
    models = {name: MODELS[name] for name in args.models}
    cutoffs = sorted(set(args.trees)) if args.trees else None
    assert args.cap > max(SELECT_K), "cap must exceed the largest evaluated K"
    t_start = time.time()
    log = lambda m: print(f"[{time.time() - t_start:7.1f}s | peak {peak_mb():6,.0f} MB] {m}", flush=True)
    cfg = Config.discover()

    split = pq.read_table(cfg.work_dir / "cache" / "cv_v1" / "train_s1_split.parquet").to_pandas()
    q_ids, n_train = holdout_queries(split, args.n)
    full = args.n is None or len(q_ids) >= 10_000
    out_dir = cfg.work_dir / "experiments" / "ranker_fast_eval"
    if args.out_name:
        out_dir = out_dir / args.out_name
    elif not full:
        out_dir = out_dir / f"smoke_n{len(q_ids)}"
    clash = [f for f in RESULT_FILES if (out_dir / f).exists()]
    if clash:
        raise SystemExit(f"refusing to overwrite existing results in {out_dir}: {clash}")
    info = split.set_index("entity_id").loc[q_ids]
    assert not info["dev"].any() and (info["fold"] != 0).all(), "evaluation queries must not come from fold 0"
    log(f"{len(q_ids):,} holdout S1 selected (of the {n_train:,} training S1), cap {args.cap}, chunk {args.chunk}")

    boosters, model_info = load_models(cfg.work_dir / "experiments" / "ranker", models, log)

    ctx = Context(cfg, q_ids, log)
    rows = pd.Index(ctx.q_ids).get_indexer(q_ids)
    assert (rows >= 0).all()
    n = len(rows)
    local = np.full(len(ctx.q_ids), -1, dtype=np.int64)
    local[rows] = np.arange(n)
    tq, tt = local[ctx.truth_q], ctx.truth_t  # every loaded truth pair belongs to a selected query

    names = {f"{p}/IDF": p for p in POOLS}
    if cutoffs:
        for name in models:
            assert cutoffs[-1] <= boosters[name].num_trees(), f"{name} has only {boosters[name].num_trees()} trees"
        names.update({f"{name}@{c}": models[name][0] for name in models for c in cutoffs})
        tree_time = {f"{name}@{c}": 0.0 for name in models for c in cutoffs}
        overlap = {f"{name}@{c}": {"ref_top": 0, "shared_top": 0, "queries_identical_top": 0}
                   for name in models for c in cutoffs}
    else:
        names.update({name: pool for name, (pool, _) in models.items()})
    evs = {name: RetrievalEvaluator(n, tq, tt) for name in names}
    timing = {"retrieval_s": 0.0, "features_s": 0.0, "predict_s": 0.0}
    pairs = {p: {"all": 0, "scored": 0} for p in POOLS}
    done, t_pass = 0, time.time()
    for block_rows, merged, qstats, dt in ctx.blocks(rows, args.chunk):
        timing["retrieval_s"] += dt
        lq = local[block_rows]
        for p, m in merged.items():
            keep = m.union_rank < args.cap
            pairs[p]["all"] += len(m.q)
            pairs[p]["scored"] += int(keep.sum())
            evs[f"{p}/IDF"].add(lq[m.q], m.t, m.union_rank)
            t0 = time.time()
            X = ctx.features(p, m, qstats, block_rows)[keep]
            timing["features_s"] += time.time() - t0
            q_k, t_k = m.q[keep], m.t[keep]
            if cutoffs:
                for name, (pool, _) in models.items():
                    if pool != p:
                        continue
                    raw, prev, cum, top = np.zeros(len(q_k)), 0, 0.0, {}
                    for c in cutoffs:
                        t0 = time.time()
                        raw += boosters[name].predict(X, raw_score=True, start_iteration=prev,
                                                      num_iteration=c - prev, num_threads=args.threads)
                        cum += time.time() - t0
                        prev = c
                        tree_time[f"{name}@{c}"] += cum  # time a standalone c-tree prediction needs
                        rank = ranker.rank_by_score(q_k, t_k, raw, len(block_rows))
                        evs[f"{name}@{c}"].add(lq[q_k], t_k, rank)
                        top[c] = rank < args.overlap_k
                    timing["predict_s"] += cum
                    ref = top[cutoffs[-1]]
                    for c in cutoffs:
                        o = overlap[f"{name}@{c}"]
                        o["ref_top"] += int(ref.sum())
                        o["shared_top"] += int((top[c] & ref).sum())
                        diff = np.bincount(q_k[top[c] ^ ref], minlength=len(block_rows))
                        o["queries_identical_top"] += int((diff == 0).sum())
                continue
            for name, pool in names.items():
                if pool != p or name.endswith("/IDF"):
                    continue
                t0 = time.time()
                score = boosters[name].predict(X, num_threads=args.threads)
                timing["predict_s"] += time.time() - t0
                evs[name].add(lq[q_k], t_k, ranker.rank_by_score(q_k, t_k, score, len(block_rows)))
        done += len(block_rows)
        el = time.time() - t_pass
        log(f"progress {done:,}/{n:,} queries ({100 * done / n:.1f}%), elapsed {el:.0f}s, "
            f"ETA {el / done * (n - done):.0f}s | retrieval {timing['retrieval_s']:.0f}s "
            f"features {timing['features_s']:.0f}s predict {timing['predict_s']:.0f}s")
    timing = {k: round(v, 1) for k, v in timing.items()}
    timing["pass_s"] = round(time.time() - t_pass, 1)

    # ---- metrics
    n_true = np.bincount(tq, minlength=n)
    q_country = ctx.q_country[rows]
    q_bucket = match_bucket(n_true)
    strata = {f"country={c}": q_country == c for c in np.unique(q_country)}
    strata.update({f"bucket={b}": q_bucket == b for b in ("0", "1", "2-3", "4+")})
    results = {}
    for name, ev in evs.items():
        results[name] = {"curve": curve(ev),
                         "by_stratum": {str(k): ev.summary(k=k, query_strata=strata)["by_query_stratum"]
                                        for k in SELECT_K}}
    ceiling = {}
    for p in POOLS:
        found = evs[f"{p}/IDF"].retrieved()
        within = evs[f"{p}/IDF"].retrieved(args.cap)
        ceiling[p] = {"true_pairs": int(len(found)), "retrieved_any_rank": int(found.sum()),
                      "retrieved_within_cap": int(within.sum()),
                      "lost_to_cap": int(found.sum() - within.sum()),
                      "pair_recall_any_rank": round(float(found.mean()), 4),
                      "pair_recall_within_cap": round(float(within.mean()), 4),
                      "pair_recall_by_union_rank_cap": {str(c): round(float(evs[f"{p}/IDF"].retrieved(c).mean()), 4)
                                                        for c in CAP_PROBES}}
    chosen = {}
    for p in POOLS if not cutoffs else ():
        score = {name: float(np.mean([results[name]["curve"][str(k)]["oracle_f05"] for k in SELECT_K]))
                 for name, (pool, _) in models.items() if pool == p}
        if not score:
            continue
        best = max(score, key=score.get)
        chosen[p] = {"config": best.split("/")[1], "mean_oracle_f05_over_K": {k: round(v, 4) for k, v in score.items()}}
        log(f"choice for {p}: {chosen[p]['config']} {chosen[p]['mean_oracle_f05_over_K']}")

    # ---- outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.DataFrame({"country": q_country, "bucket": q_bucket}).value_counts().sort_index()
    meta = {
        "script": "e03_ranker_fast_eval.py", "args": vars(args), "n_queries": n, "n_true_pairs": int(len(tq)),
        "query_set": "e03_ranker early-stopping holdout (10% hash of the 100k stratified training S1, folds 1-4)"
                     + ("" if full else f", stratified subset of {n}"),
        "sample_strata": {f"{c}|{b}": int(v) for (c, b), v in sample.items()},
        "models": model_info, "choice": chosen, "cap_ceiling": ceiling, "pairs": pairs, "timing": timing,
        "index_build_s": round(ctx.index_seconds, 1), "peak_working_set_mb": round(peak_mb()),
        "total_s": round(time.time() - t_start, 1),
    }
    if cutoffs:
        meta["tree_predict_s"] = {k: round(v, 1) for k, v in tree_time.items()}
        meta["top_k_overlap_vs_largest_cutoff"] = {
            k: dict(o, overlap_k=args.overlap_k, reference=f"{k.split('@')[0]}@{cutoffs[-1]}",
                    shared_fraction=round(o["shared_top"] / max(o["ref_top"], 1), 4),
                    pct_queries_identical_top=round(100 * o["queries_identical_top"] / n, 2))
            for k, o in overlap.items()}
    (out_dir / "fast_eval_results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    fields = ["model", "K", "mean_candidates", "pair_recall", "s1_recall", "oracle_f05", "pct_all_true_retrieved"]
    with open(out_dir / "fast_eval_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for name, r in results.items():
            for k in SELECT_K:
                c = r["curve"][str(k)]
                w.writerow([name, k] + [c[x] for x in fields[2:]])
    log(f"wrote {out_dir}")
    for name, r in results.items():
        print(f"{name:16s} " + "  ".join(
            f"K={k}: {r['curve'][str(k)]['mean_candidates']:.1f}/{r['curve'][str(k)]['pair_recall']:.4f}/"
            f"{r['curve'][str(k)]['s1_recall']:.4f}/{r['curve'][str(k)]['oracle_f05']:.4f}/"
            f"{r['curve'][str(k)]['pct_all_true_retrieved']:.1f}%" for k in SELECT_K))
    return 0


if __name__ == "__main__":
    sys.exit(main())
