"""Score a range of the ``predict`` block plan in a separate process (extra worker / benchmark).

Uses ``predict.Index``, ``Index.merged_blocks`` and ``predict.score_block`` unchanged, with the
full plan of the reference run (same chunk, same per-country row order), so a block's
checkpoint does not depend on which process scores it. Checkpoints go to this run's own
``blocks`` folder (atomic ``.tmp`` + replace, resumable); the reference run is only read.

    # benchmark: plan blocks [0, 4) with per-phase timings, compared to the reference checkpoints
    python -m ber.score_blocks --ref-run full --run-name bench_w1 --start 0 --stop 4 --profile --compare
    # extra worker: plan blocks [2600, 3466)
    python -m ber.score_blocks --ref-run full --run-name worker_tail --start 2600 --stop 3466
"""

import argparse
import functools
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import matcher, predict, ranker
from .config import Config

PHASES = defaultdict(float)


def _timed(owner, name, phase):
    fn = getattr(owner, name)

    @functools.wraps(fn)
    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            PHASES[phase] += time.perf_counter() - t
    setattr(owner, name, wrapper)


def install_profiler(models):
    """Per-phase wall time (this process only; values and call order are unchanged)."""
    _timed(predict.ChannelRunner, "run", "retrieval")
    _timed(predict.tokens, "query_matrix", "retrieval (query matrices)")
    _timed(ranker, "merge_channels", "retrieval merge")
    _timed(ranker, "query_block_stats", "query stats")
    _timed(matcher, "cap_merged", "cap 500")
    _timed(ranker, "pair_features", "stage-1 features")
    _timed(models.stage1, "predict", "stage-1 LightGBM")
    _timed(ranker, "rank_by_score", "stage-1 rank/top-50")
    _timed(matcher, "stage2_features", "stage-2 features")
    _timed(models.stage2, "predict", "stage-2 LightGBM")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--ref-run", default="full", help="run whose run_config.json / plan this worker follows")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--start", type=int, required=True, help="first plan position (inclusive)")
    ap.add_argument("--stop", type=int, required=True, help="last plan position (exclusive)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--compare", action="store_true", help="compare each block to the reference checkpoint")
    args = ap.parse_args(argv)
    predict.disable_ground_truth()
    t_start = time.time()
    log = lambda m: print(f"[{time.time() - t_start:7.1f}s] {m}", flush=True)
    cfg = Config.discover()
    ref_dir = cfg.work_dir / "inference" / args.split / args.ref_run
    run_dir = cfg.work_dir / "inference" / args.split / args.run_name
    blocks_dir = run_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    models = predict.Models(cfg)
    ref_cfg = json.loads((ref_dir / "run_config.json").read_text(encoding="utf-8"))
    fp = dict(models.fingerprint(), split=args.split, sample=None, chunk=ref_cfg["chunk"])
    if json.loads(json.dumps(fp)) != ref_cfg:
        raise SystemExit("models/configuration differ from the reference run's run_config.json")
    (run_dir / "run_config.json").write_text(json.dumps(dict(fp, ref_run=args.ref_run, start=args.start,
                                                             stop=args.stop), indent=1), encoding="utf-8")
    if args.profile:
        install_profiler(models)

    index = predict.Index(cfg, args.split, log)
    plan = index.block_plan(predict.select_rows(index), ref_cfg["chunk"])
    mine = {k for k, _, _ in plan[args.start:args.stop]}
    done = {p.stem for p in blocks_dir.glob("*.parquet")}
    skip = {k for k, _, _ in plan} - (mine - done)
    n_todo = sum(len(b) for k, _, b in plan if k not in skip)
    log(f"plan {len(plan):,} blocks; this worker: [{args.start}, {args.stop}) = {len(mine):,} blocks, "
        f"{len(mine & done):,} already done, {n_todo:,} S1 to score, threads {args.threads}")
    if args.profile:
        PHASES.clear()  # index build is not part of the per-block profile

    t_pass, processed, mismatches = time.time(), 0, 0
    t_block = time.perf_counter()
    for key, brows, m, qstats in index.merged_blocks(plan, skip=skip):
        out = predict.score_block(index, models, brows, m, qstats, args.threads)
        table = pa.table({c: out[c] for c in predict.CHECKPOINT_COLUMNS})
        tmp = blocks_dir / f"{key}.parquet.tmp"
        pq.write_table(table, tmp)
        os.replace(tmp, blocks_dir / f"{key}.parquet")
        processed += len(brows)
        now = time.perf_counter()
        block_s, t_block = now - t_block, now
        el = time.time() - t_pass
        msg = (f"block {key}: {len(brows)} S1, {len(out['p']):,} pairs, {1000 * block_s / len(brows):.1f} ms/S1 | "
               f"{processed:,}/{n_todo:,} S1, elapsed {el:.0f}s, avg {1000 * el / processed:.1f} ms/S1, "
               f"ETA {el / processed * (n_todo - processed):.0f}s")
        if args.compare:
            ref = ref_dir / "blocks" / f"{key}.parquet"
            if ref.exists():
                r = pq.read_table(ref)
                same = r.schema == table.schema and all(
                    np.array_equal(r[c].to_numpy(), table[c].to_numpy()) for c in predict.CHECKPOINT_COLUMNS)
                mismatches += not same
                msg += f" | identical to {args.ref_run}: {same}"
            else:
                msg += " | no reference checkpoint"
        log(msg)
    total = time.time() - t_pass
    if args.profile and processed:
        log("per-phase wall time (ms/S1): " + ", ".join(
            f"{k} {1000 * v / processed:.1f}" for k, v in sorted(PHASES.items(), key=lambda kv: -kv[1]))
            + f"; total {1000 * total / processed:.1f}")
    if args.compare:
        log(f"compare: {mismatches} mismatching blocks")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
