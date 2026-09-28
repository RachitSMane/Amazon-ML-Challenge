"""E03b: stage-1 LightGBM ranker on the G (and G+Indic-key) retrieval pools.

Protocol (no information from fold 0 is used before the final report):

1. Training S1: 100k from folds 1-4 (stratified); 10% of them (by S1 hash) form a holdout.
2. Training pass: retrieve G and G+I for the training S1, label pairs with the ground truth,
   keep all positives and a sample of negatives (hard = union IDF rank < 200 at 5%, the rest
   at 0.5%), store features.
3. Train three imbalance configurations per pool: ``unweighted`` (sampled rows as they are),
   ``weighted`` (negatives weighted by 1 / sampling rate = the real candidate distribution),
   ``hard_only`` (positives + hard negatives, unweighted). Early stopping on the holdout.
4. Holdout pass: score **all** candidates of the holdout S1 with every model; pick per pool
   the configuration with the best mean oracle F0.5 over K in {20, 50, 100, 200}.
5. Dev pass (fold-0 subset, 100k S1): compare IDF ranking vs LightGBM ranking for G and G+I.

Outputs: ``work/experiments/ranker/``. Run from code/business_entity_resolution/src:

    python ../experiments/e03_ranker.py [--train 100000] [--dev-limit N]
"""

import argparse
import json
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

from e02_retrieval import peak_mb, script_group  # noqa: E402
from ber import io, ranker, tokens  # noqa: E402
from ber.config import Config  # noqa: E402
from ber.retrieval.channels import C1, C2, K, Channel, ChannelRunner  # noqa: E402
from ber.retrieval.evaluate import RetrievalEvaluator  # noqa: E402
from ber.retrieval.indic import INDIC_KEY_FIELD, build_indic_key_pools, skeleton_keys  # noqa: E402

K_VALUES = (5, 10, 20, 50, 100, 200)
SELECT_K = (20, 50, 100, 200)
G = {"name": C1(10000), "addr": C2(1000), "key": K()}
GI = dict(G, ikey=Channel("indic_key", INDIC_KEY_FIELD))
POOLS = {"G": tuple(G), "GI": tuple(GI)}
CONFIGS = ("unweighted", "weighted", "hard_only")
QUERY_COLUMNS = ["entity_id", "country", "name_fold", "address_norm", "name_key", "name_stripped", "name_script",
                 "address_missing", "house_numbers", "unit_numbers", "postcode", "source"]
TARGET_COLUMNS = ["country", "name_script", "name_stripped", "address_missing", "house_numbers", "unit_numbers",
                  "postcode", "source"]


class Context:
    """Everything shared by the passes: pools, record arrays, truth, query tables."""

    def __init__(self, cfg, s1_ids, log):
        self.log = log
        s1 = tokens.read_normalized(cfg, "train", (1,), QUERY_COLUMNS)
        self.s1 = s1.filter(pc.is_in(s1["entity_id"], value_set=pa.array(list(s1_ids))))
        self.q_ids = self.s1["entity_id"].to_pandas().astype(str)
        self.q_country = self.s1["country"].to_numpy(zero_copy_only=False)
        self.queries = ranker.RecordArrays.from_table(self.s1)
        self.q_ikey = pa.array(skeleton_keys(self.s1["name_stripped"].to_pylist()), pa.string())
        log(f"{len(self.s1):,} queries loaded")

        t0 = time.time()
        self.pool = tokens.build_target_pool(cfg, "train", fields=("name_fold", "address_norm", "name_key"))
        tt = tokens.read_normalized(cfg, "train", (2, 3), TARGET_COLUMNS)
        self.targets = ranker.RecordArrays.from_table(tt)
        self.target_script = script_group(tt["name_script"].to_numpy(zero_copy_only=False))
        self.ikey_pools = build_indic_key_pools(tt["name_stripped"].to_pylist(),
                                                tt["country"].to_numpy(zero_copy_only=False),
                                                tt["name_script"].to_numpy(zero_copy_only=False), list(self.pool.countries))
        del tt
        ranker.add_target_statistics(self.targets, self.pool, GI, extra_pools={"ikey": self.ikey_pools})
        self.index_seconds = time.time() - t0
        log(f"target pool, record arrays and Indic-key index built in {self.index_seconds:.0f}s")

        gt = io.load_ground_truth(cfg).pairs
        gt = gt[gt["source1_entity_id"].isin(set(self.q_ids))]
        self.truth_q = pd.Index(self.q_ids).get_indexer(gt["source1_entity_id"].astype(str))
        self.truth_t = pd.Index(self.pool.entity_id.to_pandas().astype(str)).get_indexer(gt["target_entity_id"].astype(str))
        assert (self.truth_q >= 0).all() and (self.truth_t >= 0).all()
        self.truth_keys = np.sort(ranker.pair_keys(self.truth_q, self.truth_t))

    def blocks(self, query_rows, chunk):
        """Yield (global query rows of the block, merged G, merged GI, query stats) per chunk."""
        query_rows = np.asarray(query_rows)
        for c, cpool in self.pool.countries.items():
            rows = query_rows[self.q_country[query_rows] == c]
            if not len(rows):
                continue
            sub = self.s1.take(pa.array(rows))
            main = ChannelRunner(cpool, {f: tokens.query_matrix(sub[f], tokens.FIELD_MODES[f], cpool.fields[f])
                                         for f in ("name_fold", "address_norm", "name_key")})
            icp = self.ikey_pools[c]
            irun = ChannelRunner(icp, {INDIC_KEY_FIELD: tokens.query_matrix(self.q_ikey.take(pa.array(rows)), "key",
                                                                             icp.fields[INDIC_KEY_FIELD])})
            runners = {"name": main, "addr": main, "key": main, "ikey": irun}
            for start in range(0, len(rows), chunk):
                stop = min(start + chunk, len(rows))
                t0 = time.time()
                parts = {p: runners[p].run(ch, start, stop) for p, ch in GI.items()}
                mg = ranker.merge_channels([parts[p] for p in POOLS["G"]], stop - start)
                mgi = ranker.merge_channels([parts[p] for p in POOLS["GI"]], stop - start)
                qstats = ranker.query_block_stats(main, runners, GI, start, stop)
                yield rows[start:stop], {"G": mg, "GI": mgi}, qstats, time.time() - t0

    def features(self, pool_name, m, qstats, block_rows):
        return ranker.pair_features(m, POOLS[pool_name], qstats, block_rows, self.queries, self.targets)


def collect_training(ctx, rows, chunk, seed=0):
    rng = np.random.default_rng(seed)
    data = {p: {"X": [], "y": [], "w": [], "hard": [], "q": []} for p in POOLS}
    counts = {p: {"candidate_pairs": 0, "positives": 0} for p in POOLS}
    t_ret = t_feat = 0.0
    for block_rows, merged, qstats, dt in ctx.blocks(rows, chunk):
        t_ret += dt
        t0 = time.time()
        for p, m in merged.items():
            qg = block_rows[m.q]
            y = ranker.labels_for(qg, m.t, ctx.truth_keys)
            sel, w, hard = ranker.sample_negatives(y, m.union_rank, rng)
            X = ctx.features(p, m, qstats, block_rows)[sel]
            d = data[p]
            d["X"].append(X); d["y"].append(y[sel]); d["w"].append(w); d["hard"].append(hard); d["q"].append(qg[sel])
            counts[p]["candidate_pairs"] += len(y)
            counts[p]["positives"] += int(y.sum())
        t_feat += time.time() - t0
    out = {p: {k: np.concatenate(v) for k, v in d.items()} for p, d in data.items()}
    for p in POOLS:
        y, hard = out[p]["y"], out[p]["hard"]
        counts[p].update({"sampled_rows": int(len(y)), "sampled_positives": int(y.sum()),
                          "sampled_hard_negatives": int(((y == 0) & hard).sum()),
                          "sampled_rest_negatives": int(((y == 0) & ~hard).sum()),
                          "true_pairs_of_training_s1": int(np.isin(ctx.truth_q, rows).sum())})
    return out, counts, t_ret, t_feat


def train_models(ctx, data, rows, holdout_rows):
    models, info = {}, {}
    hold_set = set(holdout_rows.tolist())
    for p in POOLS:
        d = data[p]
        is_hold = np.fromiter((q in hold_set for q in d["q"]), bool, count=len(d["q"]))
        names = ranker.feature_names(POOLS[p])
        for cfg_name in CONFIGS:
            if cfg_name == "hard_only":
                keep = (d["y"] == 1) | d["hard"]
                w = np.ones(len(d["y"]), np.float32)
            else:
                keep = np.ones(len(d["y"]), bool)
                w = d["w"] if cfg_name == "weighted" else np.ones(len(d["y"]), np.float32)
            fit, val = keep & ~is_hold, keep & is_hold
            t0 = time.time()
            booster, evals = ranker.train_lgbm(d["X"][fit], d["y"][fit], w[fit], d["X"][val], d["y"][val], w[val], names)
            gain = booster.feature_importance("gain")
            models[(p, cfg_name)] = booster
            info[f"{p}/{cfg_name}"] = {
                "train_rows": int(fit.sum()), "train_positives": int(d["y"][fit].sum()),
                "valid_rows": int(val.sum()), "best_iteration": int(booster.best_iteration),
                "valid_logloss": round(float(evals["val"]["binary_logloss"][booster.best_iteration - 1]), 5),
                "valid_auc": round(float(evals["val"]["auc"][booster.best_iteration - 1]), 5),
                "seconds": round(time.time() - t0, 1),
                "top_features_by_gain": [[names[i], round(float(gain[i] / gain.sum()), 4)] for i in np.argsort(-gain)[:15]],
            }
            ctx.log(f"trained {p}/{cfg_name}: {info[f'{p}/{cfg_name}']['best_iteration']} trees, "
                    f"AUC {info[f'{p}/{cfg_name}']['valid_auc']}, {info[f'{p}/{cfg_name}']['seconds']}s")
    return models, info


def pass_truth(ctx, rows):
    """Local query index for ``rows`` and the true pairs of those queries (local q, target, mask)."""
    local = np.full(len(ctx.q_ids), -1, dtype=np.int64)
    local[rows] = np.arange(len(rows))
    mask = local[ctx.truth_q] >= 0
    return local, local[ctx.truth_q[mask]], ctx.truth_t[mask], mask


def evaluate_pass(ctx, rows, chunk, rankers):
    """Score all candidates of ``rows`` (evaluators indexed by position in ``rows``).

    ``rankers``: {name: (pool, booster or None for the IDF union rank)}.
    """
    local, tq, tt, _ = pass_truth(ctx, rows)
    evs = {name: RetrievalEvaluator(len(rows), tq, tt) for name in rankers}
    t_ret = t_feat = t_pred = 0.0
    n_pairs = {p: 0 for p in POOLS}
    for block_rows, merged, qstats, dt in ctx.blocks(rows, chunk):
        t_ret += dt
        X = {}
        for p in POOLS:
            t0 = time.time()
            X[p] = ctx.features(p, merged[p], qstats, block_rows)
            n_pairs[p] += len(merged[p].q)
            t_feat += time.time() - t0
        for name, (p, booster) in rankers.items():
            m = merged[p]
            t0 = time.time()
            if booster is None:
                rank = m.union_rank
            else:
                rank = ranker.rank_by_score(m.q, m.t, booster.predict(X[p], num_threads=8), len(block_rows))
            t_pred += time.time() - t0
            evs[name].add(local[block_rows[m.q]], m.t, rank)
    return evs, {"retrieval_s": round(t_ret, 1), "features_s": round(t_feat, 1), "predict_s": round(t_pred, 1),
                 "pairs": n_pairs}


def curve(ev):
    out = {}
    for k in (None,) + K_VALUES:
        s = ev.summary(k=k)
        out["all" if k is None else str(k)] = {
            "mean_candidates": s["candidates"]["mean"], "p50": s["candidates"]["median"], "p90": s["candidates"]["p90"],
            "p95": s["candidates"]["p95"], "p99": s["candidates"]["p99"], "pair_recall": s["pair_recall"],
            "s1_recall": s["s1_recall"], "oracle_f05": s["oracle_f05"], "pct_all_true_retrieved": s["pct_all_true_retrieved"]}
    return out


def strata_topk(ev, strata, k):
    found = ev.retrieved(k)
    return {name: round(float(found[m].mean()), 4) for name, m in strata.items() if m.any()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=100_000)
    ap.add_argument("--dev-limit", type=int)
    ap.add_argument("--chunk", type=int, default=1000)
    args = ap.parse_args(argv)
    t_start = time.time()
    log = lambda m: print(f"[{time.time() - t_start:7.1f}s | peak {peak_mb():6,.0f} MB] {m}", flush=True)
    cfg = Config.discover()
    out_dir = cfg.work_dir / "experiments" / "ranker"
    out_dir.mkdir(parents=True, exist_ok=True)

    split = pq.read_table(cfg.work_dir / "cache" / "cv_v1" / "train_s1_split.parquet").to_pandas()
    dev_ids = split.loc[split["dev"], "entity_id"].to_numpy()
    if args.dev_limit:
        dev_ids = dev_ids[: args.dev_limit]
    train_ids = ranker.select_training_s1(split, args.train)
    assert not set(train_ids) & set(split.loc[split["fold"] == 0, "entity_id"]), "training S1 from fold 0"
    hold = ranker.holdout_mask(train_ids)
    ctx = Context(cfg, np.concatenate([train_ids, dev_ids]), log)
    index = pd.Index(ctx.q_ids)
    train_rows = index.get_indexer(train_ids)
    dev_rows = index.get_indexer(dev_ids)
    holdout_rows = train_rows[hold]
    assert (train_rows >= 0).all() and (dev_rows >= 0).all()
    assert not set(train_rows) & set(dev_rows)
    log(f"train S1 {len(train_rows):,} (holdout {len(holdout_rows):,}), dev S1 {len(dev_rows):,}")

    # ---- training data and models
    data, counts, t_ret, t_feat = collect_training(ctx, train_rows, args.chunk)
    log(f"training rows collected (retrieval {t_ret:.0f}s, features+sampling {t_feat:.0f}s): {counts}")
    feat_stats = {}
    for p in POOLS:
        names = ranker.feature_names(POOLS[p])
        X, y = data[p]["X"], data[p]["y"]
        feat_stats[p] = {n: {"mean_pos": round(float(np.nanmean(X[y == 1, j])), 4),
                             "mean_neg": round(float(np.nanmean(X[y == 0, j])), 4),
                             "nan_share": round(float(np.isnan(X[:, j]).mean()), 4)} for j, n in enumerate(names)}
    models, model_info = train_models(ctx, data, train_rows, holdout_rows)
    del data
    for (p, c), booster in models.items():
        booster.save_model(str(out_dir / f"model_{p}_{c}.txt"))

    # ---- holdout pass: choose the configuration per pool (fold 0 not involved)
    rankers = {f"{p}/IDF": (p, None) for p in POOLS}
    rankers.update({f"{p}/{c}": (p, models[(p, c)]) for p in POOLS for c in CONFIGS})
    evs, hold_timing = evaluate_pass(ctx, holdout_rows, args.chunk, rankers)
    holdout = {name: curve(ev) for name, ev in evs.items()}
    chosen = {}
    for p in POOLS:
        score = {c: np.mean([holdout[f"{p}/{c}"][str(k)]["oracle_f05"] for k in SELECT_K]) for c in CONFIGS}
        chosen[p] = max(score, key=score.get)
        log(f"holdout choice for {p}: {chosen[p]} ({ {c: round(v, 4) for c, v in score.items()} })")

    # ---- dev pass: final comparison
    rankers = {"G raw / IDF": ("G", None), "G+I raw / IDF": ("GI", None)}
    rankers.update({f"G LightGBM {c}": ("G", models[("G", c)]) for c in CONFIGS})
    rankers[f"G+I LightGBM {chosen['GI']}"] = ("GI", models[("GI", chosen["GI"])])
    t0 = time.time()
    evs, dev_timing = evaluate_pass(ctx, dev_rows, args.chunk, rankers)
    dev_timing["total_s"] = round(time.time() - t0, 1)
    _, _, _, dev_pair = pass_truth(ctx, dev_rows)
    tq, tt = ctx.truth_q[dev_pair], ctx.truth_t[dev_pair]  # same order as the evaluators' pairs
    strata = {
        "country=US": ctx.q_country[tq] == "US", "country=India": ctx.q_country[tq] == "India",
        "target=S2": ctx.pool.source[tt] == 2, "target=S3": ctx.pool.source[tt] == 3,
        "target Indic": ctx.target_script[tt] == "indic", "target Latin": ctx.target_script[tt] == "latin",
        "target address missing": ctx.pool.address_missing[tt],
    }
    results = {name: {"curve": curve(ev), "strata_top_k": {str(k): strata_topk(ev, strata, k)
                                                           for k in (20, 50, 100, 200, None)}}
               for name, ev in evs.items()}
    meta = {
        "n_train_s1": int(len(train_rows)), "n_holdout_s1": int(len(holdout_rows)), "n_dev_s1": int(len(dev_rows)),
        "training_counts": counts, "models": model_info, "holdout_choice": chosen,
        "index_build_s": round(ctx.index_seconds, 1), "holdout_timing": hold_timing, "dev_timing": dev_timing,
        "peak_working_set_mb": round(peak_mb()), "total_s": round(time.time() - t_start, 1),
        "lgb_params": ranker.LGB_PARAMS, "sampling": {"hard_rank": 200, "hard_rate": 0.05, "rest_rate": 0.005},
    }
    for name, obj in (("dev_results", results), ("holdout_results", holdout), ("feature_stats", feat_stats), ("run_meta", meta)):
        (out_dir / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    log(f"wrote {out_dir}")
    for name, r in results.items():
        line = "  ".join(f"K={k}: {r['curve'][k]['mean_candidates']:.1f}/{r['curve'][k]['pair_recall']:.4f}/{r['curve'][k]['oracle_f05']:.4f}"
                         for k in ("20", "50", "100", "200", "all"))
        print(f"{name:28s} {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
