"""Fallback submission from the checkpoints finished so far (no scoring, no model changes).

Rebuilds only what ``predict.finalize`` needs (S1 IDs/countries and target IDs/sources, in
the same order as ``predict.Index``) from the normalized cache, restricts the block plan to
the blocks that already have a checkpoint, and calls ``predict.finalize`` unchanged: the same
threshold, the same one-target -> one-S1 decision over all finished blocks, and one row per
S1 of the split (unscored S1 get empty lists). Checkpoints are only read.

Run from code/business_entity_resolution/src (never point --out at the real output folder
while the full run is still going):

    python -m ber.assemble_partial --split test --run-name full --out ../../../work/inference/test/fallback_preview --validate
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.compute as pc

from . import predict, tokens
from .config import REPO_ROOT, Config


class LightIndex:
    """The fields of ``predict.Index`` that ``finalize`` and ``block_plan`` read, without retrieval."""

    def __init__(self, cfg, split):
        self.split = split
        s1 = tokens.read_normalized(cfg, split, (1,), ["entity_id", "country"])
        self.s1_ids = s1["entity_id"].combine_chunks()
        self.q_country = s1["country"].to_numpy(zero_copy_only=False)
        tt = tokens.read_normalized(cfg, split, (2, 3), ["entity_id", "country", "source"])
        countries = sorted(pc.unique(tt["country"]).to_pylist())  # same order as build_target_pool
        self.pool = SimpleNamespace(entity_id=tt["entity_id"].combine_chunks(), source=tt["source"].to_numpy(),
                                    countries=dict.fromkeys(countries))

    block_plan = predict.Index.block_plan


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out", required=True, help="folder for the two preview TSV files")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args(argv)
    predict.disable_ground_truth()
    t_start = time.time()
    log = lambda m: print(f"[{time.time() - t_start:7.1f}s] {m}", flush=True)
    cfg = Config.discover()
    run_dir = cfg.work_dir / "inference" / args.split / args.run_name
    out_dir = Path(args.out).resolve()
    if out_dir == (REPO_ROOT / "output").resolve():
        raise SystemExit("refusing to overwrite the real output folder")

    models = predict.Models(cfg)
    saved = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    for k, v in models.fingerprint().items():
        if json.loads(json.dumps(v)) != saved[k]:
            raise SystemExit(f"{k} differs from the run's run_config.json")

    index = LightIndex(cfg, args.split)
    full_plan = index.block_plan(np.arange(len(index.s1_ids)), saved["chunk"])
    done = {p.stem for p in (run_dir / "blocks").glob("*.parquet")}  # snapshot; .tmp files are excluded
    plan = [b for b in full_plan if b[0] in done]
    if len(plan) != len(done):
        raise SystemExit(f"checkpoints outside the plan: {sorted(done - {k for k, _, _ in full_plan})[:3]}")
    log(f"{len(plan):,}/{len(full_plan):,} blocks finished ({sum(len(b) for _, _, b in plan):,}/"
        f"{len(index.s1_ids):,} S1)")

    summary = predict.finalize(index, models, plan, run_dir, out_dir, log)
    summary.update(blocks_done=len(plan), blocks_total=len(full_plan), total_s=round(time.time() - t_start, 1))
    if args.validate:
        res = subprocess.run([sys.executable, str(REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"),
                              "--matching", str(out_dir / "matching_results.tsv"),
                              "--candidate", str(out_dir / "candidate_pairs.tsv"),
                              "--test-dir", str(cfg.data_dir / args.split), "--check-ids"],
                             capture_output=True, text=True)
        log(f"validator exit {res.returncode}:\n{res.stdout}{res.stderr}")
        summary["validator_exit"] = res.returncode
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
