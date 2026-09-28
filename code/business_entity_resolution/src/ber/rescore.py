"""Re-score a finished run's stage-1 top-K with a different stage-2 model (no stage-1 LightGBM).

A reference run's checkpoints hold, per block, the stage-1 top-K pairs with their raw
stage-1 score and rank -- exactly what ``matcher.stage2_features`` consumes besides the
stage-1 features. So for each block this recomputes retrieval, the cap and the stage-1
features (``predict.Index.merged_blocks`` / ``ranker.pair_features``, unchanged), locates the
checkpointed pairs among the capped rows, rebuilds the 62 stage-2 features and applies the
new stage-2 model. Stage 1 (retrieval, 400 trees, cap, top-K) is therefore identical to the
reference run by construction; only stage 2 changes. ``--stage2-dir`` pointing at the
reference run's own model must reproduce its ``p`` exactly (``--compare``).

    python -m ber.rescore --ref-run full --run-name v2 --stage2-dir <dir> [--start A --stop B] [--compare]
    python -m ber.rescore --ref-run full --run-name v2 --stage2-dir <dir> --finalize --out ../../../output_v2 --validate
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import matcher, predict, ranker
from .assemble_partial import LightIndex
from .config import REPO_ROOT, Config


def rescore_block(index, models, brows, m, qstats, ref, threads=None):
    """Stage-2 ``p`` for the checkpointed top-K pairs ``ref`` of one block (same row order)."""
    mc, _ = matcher.cap_merged(m, models.stage1_cfg["cap"])
    X1 = ranker.pair_features(mc, tuple(predict.GI_CHANNELS), qstats, brows, index.queries, index.targets)
    keys = ranker.pair_keys(brows[mc.q], mc.t)
    order = np.argsort(keys, kind="stable")
    want = ranker.pair_keys(ref["s1_row"], ref["target_row"])
    pos = np.searchsorted(keys, want, sorter=order)
    top = order[np.minimum(pos, len(order) - 1)]
    if not np.array_equal(keys[top], want) or not (np.diff(top) > 0).all():
        raise RuntimeError("checkpointed pairs are not the capped candidates in stage-1 order")
    q_blk, t = mc.q[top], mc.t[top]
    s1_row = brows[q_blk]
    raw = ref["s1_score"].astype(np.float32)
    rank = ref["s1_rank"].astype(np.float32)
    qstr = {c: a.take(pa.array(s1_row)) for c, a in index.qstr.items()}
    tstr = {c: a.take(pa.array(t)) for c, a in index.tstr.items()}
    X = matcher.stage2_features(X1[top], models.stage1_names, raw, rank, q_blk, len(brows), qstr, tstr)
    p = models.stage2.predict(X, num_threads=threads or 0) if len(X) else np.zeros(0)
    return {"s1_row": ref["s1_row"], "target_row": ref["target_row"], "s1_rank": ref["s1_rank"],
            "s1_score": ref["s1_score"], "p": p.astype(np.float32)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--ref-run", default="full")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--stage2-dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=10 ** 9)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--compare", action="store_true", help="compare p with the reference run's checkpoints")
    ap.add_argument("--finalize", action="store_true", help="write the two TSVs from this run's checkpoints")
    ap.add_argument("--out", help="output folder for --finalize")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args(argv)
    predict.disable_ground_truth()
    t_start = time.time()
    log = lambda m: print(f"[{time.time() - t_start:7.1f}s] {m}", flush=True)
    cfg = Config.discover()
    ref_dir = cfg.work_dir / "inference" / args.split / args.ref_run
    run_dir = cfg.work_dir / "inference" / args.split / args.run_name
    blocks_dir = run_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    models = predict.Models(cfg, args.stage2_dir)
    ref_cfg = json.loads((ref_dir / "run_config.json").read_text(encoding="utf-8"))
    if ref_cfg["stage1"] != models.stage1_cfg or ref_cfg["stage1_sha256"] != models.stage1_sha:
        raise SystemExit("stage 1 differs from the reference run")
    fp = dict(models.fingerprint(), split=args.split, chunk=ref_cfg["chunk"], ref_run=args.ref_run,
              stage2_dir=str(Path(args.stage2_dir).resolve()))
    fp_path = run_dir / "run_config.json"
    if fp_path.exists() and json.loads(fp_path.read_text(encoding="utf-8")) != json.loads(json.dumps(fp)):
        raise SystemExit(f"{fp_path} was written with a different configuration")
    fp_path.write_text(json.dumps(fp, indent=1), encoding="utf-8")

    if args.finalize:
        out_dir = Path(args.out).resolve()
        if out_dir == (REPO_ROOT / "output").resolve():
            raise SystemExit("refusing to write into the v1 output folder")
        index = LightIndex(cfg, args.split)
        plan = index.block_plan(np.arange(len(index.s1_ids)), ref_cfg["chunk"])
        summary = predict.finalize(index, models, plan, run_dir, out_dir, log)
        if args.validate:
            res = subprocess.run([sys.executable, str(REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"),
                                  "--matching", str(out_dir / "matching_results.tsv"),
                                  "--candidate", str(out_dir / "candidate_pairs.tsv"),
                                  "--test-dir", str(cfg.data_dir / args.split), "--check-ids"],
                                 capture_output=True, text=True)
            log(f"validator exit {res.returncode}:\n{res.stdout}{res.stderr}")
            summary["validator_exit"] = res.returncode
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        return 0

    index = predict.Index(cfg, args.split, log)
    plan = index.block_plan(predict.select_rows(index), ref_cfg["chunk"])
    mine = {k for k, _, _ in plan[args.start:args.stop]}
    done = {p.stem for p in blocks_dir.glob("*.parquet")}
    skip = {k for k, _, _ in plan} - (mine - done)
    n_todo = sum(len(b) for k, _, b in plan if k not in skip)
    log(f"plan {len(plan):,} blocks; this process [{args.start}, {min(args.stop, len(plan))}): "
        f"{len(mine & done):,} done, {n_todo:,} S1 to re-score")
    t_pass, processed, max_diff = time.time(), 0, 0.0
    for key, brows, m, qstats in index.merged_blocks(plan, skip=skip):
        rt = pq.read_table(ref_dir / "blocks" / f"{key}.parquet")
        ref = {c: rt[c].to_numpy() for c in predict.CHECKPOINT_COLUMNS}
        out = rescore_block(index, models, brows, m, qstats, ref, args.threads)
        tmp = blocks_dir / f"{key}.parquet.tmp"
        pq.write_table(pa.table({c: out[c] for c in predict.CHECKPOINT_COLUMNS}), tmp)
        os.replace(tmp, blocks_dir / f"{key}.parquet")
        processed += len(brows)
        el = time.time() - t_pass
        msg = (f"block {key}: {len(brows)} S1, {len(out['p']):,} pairs | {processed:,}/{n_todo:,} S1, "
               f"{1000 * el / processed:.1f} ms/S1, ETA {el / processed * (n_todo - processed):.0f}s")
        if args.compare:
            d = float(np.abs(out["p"] - ref["p"]).max()) if len(ref["p"]) else 0.0
            max_diff = max(max_diff, d)
            msg += f" | max |p - ref p| {d:.3g}"
        log(msg)
    if args.compare:
        log(f"compare: max |p - ref p| over all blocks {max_diff:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
