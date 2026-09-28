"""Check that ``ber.predict`` reproduces the e04_matcher.py pipeline exactly (train split).

Reference: the e04 glue (``e03_ranker.Context`` + ``e04_matcher.load_strings`` + the e04 block
loop) with the saved stage-2 model. New: ``ber.predict`` (no ground truth). Both on the same
200 S1 of the full run's 20k fold-0 dev subset. Compares candidate order, the 62 features,
stage-2 probabilities and the thresholded one-S1-per-target predictions (the latter read back
from the written matching_results.tsv).

    python ../experiments/check_predict_repro.py
"""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e03_ranker  # noqa: E402
from e04_matcher import load_strings, select_s1  # noqa: E402
from ber import matcher, predict  # noqa: E402
from ber.config import Config  # noqa: E402
from ber.cv import stratified_subset  # noqa: E402

N = 200


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
    cfg = Config.discover()
    run_dir = cfg.work_dir / "inference" / "train" / "repro_n200"
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; refusing to reuse it")
    assert predict.GI_CHANNELS == e03_ranker.GI and predict.QUERY_COLUMNS == e03_ranker.QUERY_COLUMNS
    assert predict.TARGET_COLUMNS == e03_ranker.TARGET_COLUMNS
    split = pq.read_table(cfg.work_dir / "cache" / "cv_v1" / "train_s1_split.parquet").to_pandas()
    _, _, valid = select_s1(split, 60_000, 20_000)
    strata = split.set_index("entity_id").loc[valid, "stratum"].to_numpy()
    ids = valid[stratified_subset(valid, strata, N, seed="ber-predict-repro")]
    models = predict.Models(cfg)
    log(f"{len(ids)} dev S1 of the full run; threshold {models.threshold}")

    # ---- reference: exactly the e04 per-block code
    ctx = e03_ranker.Context(cfg, ids, log)
    qstr_all, tstr_all = load_strings(cfg, ctx)
    rows = pd.Index(ctx.q_ids).get_indexer(ids)
    st1 = matcher.STAGE1
    ref = {k: [] for k in ("s1", "t", "rank", "raw", "X", "p")}
    for block_rows, merged, qstats, _ in ctx.blocks(rows, 500):
        m, _ = matcher.cap_merged(merged[st1["pool"]], st1["cap"])
        X1 = ctx.features(st1["pool"], m, qstats, block_rows)
        top, raw, rank = matcher.stage1_top(m.q, m.t, X1, models.stage1, len(block_rows))
        q_blk, t_glob = m.q[top], m.t[top]
        q_glob = block_rows[q_blk]
        qstr = {c: a.take(pa.array(q_glob)) for c, a in qstr_all.items()}
        tstr = {c: a.take(pa.array(t_glob)) for c, a in tstr_all.items()}
        X = matcher.stage2_features(X1[top], models.stage1_names, raw, rank, q_blk, len(block_rows), qstr, tstr)
        ref["s1"].append(np.asarray(ctx.q_ids)[q_glob]); ref["t"].append(ctx.pool.entity_id.take(pa.array(t_glob)).to_numpy(zero_copy_only=False))
        ref["rank"].append(rank); ref["raw"].append(raw); ref["X"].append(X); ref["p"].append(models.stage2.predict(X))
    ref = {k: np.concatenate(v) for k, v in ref.items()}
    ref_sel = matcher.decide(pd.factorize(ref["s1"])[0], pd.factorize(ref["t"])[0], ref["p"], models.threshold)
    ref_pred = set(zip(ref["s1"][ref_sel].tolist(), ref["t"][ref_sel].tolist()))
    log(f"reference: {len(ref['p']):,} pairs, {len(ref_pred)} predicted")
    del ctx, qstr_all, tstr_all
    gc.collect()

    # ---- new pipeline, ground truth disabled
    predict.disable_ground_truth()
    index = predict.Index(cfg, "train", log)
    rows = pd.Index(index.s1_ids.to_pandas()).get_indexer(ids)
    assert (rows >= 0).all()
    new = {k: [] for k in ("s1", "t", "rank", "raw", "X", "p")}
    plan = index.block_plan(rows, 500)
    for key, brows, m, qstats in index.merged_blocks(plan):
        out = predict.score_block(index, models, brows, m, qstats)
        new["s1"].append(index.s1_ids.take(pa.array(out["s1_row"])).to_numpy(zero_copy_only=False))
        new["t"].append(index.pool.entity_id.take(pa.array(out["target_row"])).to_numpy(zero_copy_only=False))
        new["rank"].append(out["s1_rank"]); new["raw"].append(out["s1_score"]); new["X"].append(out["X"]); new["p"].append(out["p"])
    new = {k: np.concatenate(v) for k, v in new.items()}
    # full production path (checkpoints + global decision + TSV writer) on the same S1
    predict.run_blocks(index, models, rows, run_dir, 500, log=log)
    summary = predict.finalize(index, models, plan, run_dir, run_dir / "output", log)
    got = pd.read_csv(run_dir / "output" / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    got = got[got["source1_entity_id"].isin(set(ids))]
    new_pred = {(s, t) for s, lst in zip(got["source1_entity_id"], got["matched_entity_ids"]) for t in lst.split(",") if t}

    # ---- compare (both sorted by S1 id, then stage-1 rank)
    o_ref = np.lexsort((ref["rank"], ref["s1"]))
    o_new = np.lexsort((new["rank"], new["s1"]))
    res = {
        "pairs_reference": int(len(ref["p"])), "pairs_new": int(len(new["p"])),
        "same_candidates_same_order": bool(len(o_ref) == len(o_new) and (ref["s1"][o_ref] == new["s1"][o_new]).all()
                                           and (ref["t"][o_ref] == new["t"][o_new]).all()
                                           and (ref["rank"][o_ref] == new["rank"][o_new]).all()),
        "features_identical": bool(np.array_equal(ref["X"][o_ref], new["X"][o_new], equal_nan=True)),
        "n_features": int(new["X"].shape[1]),
        "stage1_score_max_abs_diff": float(np.abs(ref["raw"][o_ref] - new["raw"][o_new]).max()),
        "p_max_abs_diff": float(np.abs(ref["p"][o_ref] - new["p"][o_new]).max()),
        "predicted_pairs_reference": len(ref_pred), "predicted_pairs_new": len(new_pred),
        "differing_predictions": len(ref_pred ^ new_pred),
        "s1_rows_written": summary["s1_total"], "s1_scored": summary["s1_scored"],
    }
    log(json.dumps(res, indent=1))
    (run_dir / "repro_result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    ok = (res["same_candidates_same_order"] and res["features_identical"] and res["p_max_abs_diff"] < 1e-6
          and res["differing_predictions"] == 0)
    log("CHECK 1 PASS" if ok else "CHECK 1 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
