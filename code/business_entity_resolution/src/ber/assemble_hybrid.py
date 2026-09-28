"""Hybrid submission: blocks scored by a new configuration where available, the reference run elsewhere.

For every block of the test plan the checkpoint of ``--primary`` (e.g. the cap-1000 / top-200
run) is used if it exists, otherwise the checkpoint of ``--fallback`` (the complete v1 run).
Each pair is kept if ``p >= threshold`` of the model that scored it (each model's own dev-optimal
threshold, read from its ``config.json``); then the one-target -> one-S1 rule is applied over
all kept pairs of the whole test set (highest ``p`` wins, ties to the smaller S1 row, as in
``matcher.best_s1_per_target``). ``candidate_pairs.tsv`` lists, per S1, exactly the candidates
its model scored (stage-1 order). Checkpoints are only read.

    python -m ber.assemble_hybrid --primary experiments/stage1_400_cap1000_top200/test_run \\
        --primary-stage2 ../../../work/experiments/stage1_400_cap1000_top200/t60000_v20000 \\
        --fallback inference/test/full --fallback-stage2 ../../../work/experiments/matcher/full_t60000_v20000 \\
        --out ../../../work/experiments/stage1_400_cap1000_top200/output_hybrid --validate
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import matcher, predict, ranker
from .assemble_partial import LightIndex
from .config import REPO_ROOT, Config


def write_id_lists(path, header, s1_ids, s1_row, target_ids, target_row, step=200_000):
    """Same bytes as ``predict._write_id_lists`` (pairs pre-sorted by S1 row), built ``step`` S1 at a time."""
    n = len(s1_ids)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(s1_row, minlength=n), out=offsets[1:])
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for start in range(0, n, step):
            stop = min(start + step, n)
            lo, hi = offsets[start], offsets[stop]
            off = pa.array((offsets[start:stop + 1] - lo).astype(np.int32))
            lists = pa.ListArray.from_arrays(off, target_ids.take(pa.array(target_row[lo:hi])))
            lines = pc.binary_join_element_wise(s1_ids.slice(start, stop - start), pc.binary_join(lists, ","), "\t")
            f.write("\n".join(lines.to_pylist()) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--primary", required=True, help="run folder (under the work dir) of the new configuration")
    ap.add_argument("--primary-stage2", required=True)
    ap.add_argument("--fallback", required=True, help="complete run folder (under the work dir), e.g. inference/test/full")
    ap.add_argument("--fallback-stage2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args(argv)
    predict.disable_ground_truth()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
    cfg = Config.discover()
    out_dir = Path(args.out).resolve()
    if out_dir == (REPO_ROOT / "output").resolve():
        raise SystemExit("refusing to write into the v1 output folder")
    thr = {src: float(json.loads((Path(d) / "config.json").read_text(encoding="utf-8"))["threshold"])
           for src, d in ((1, args.primary_stage2), (0, args.fallback_stage2))}
    prim, fall = cfg.work_dir / args.primary / "blocks", cfg.work_dir / args.fallback / "blocks"
    fall_cfg = json.loads((cfg.work_dir / args.fallback / "run_config.json").read_text(encoding="utf-8"))

    index = LightIndex(cfg, args.split)
    plan = index.block_plan(np.arange(len(index.s1_ids)), fall_cfg["chunk"])
    have_p = {p.stem for p in prim.glob("*.parquet")}
    have_f = {p.stem for p in fall.glob("*.parquet")}
    missing = [k for k, _, _ in plan if k not in have_p and k not in have_f]
    if missing:
        raise SystemExit(f"{len(missing)} blocks have no checkpoint in either run (e.g. {missing[:3]})")
    tables, n_prim_s1 = [], 0  # only the checkpoint columns (+ src) are kept
    for key, _, brows in plan:
        src = 1 if key in have_p else 0
        t = pq.read_table((prim if src else fall) / f"{key}.parquet")
        if len(t) and not np.isin(t["s1_row"].to_numpy(), brows).all():
            raise SystemExit(f"block {key}: pairs for S1 outside the block")
        tables.append(t.append_column("src", pa.array(np.full(len(t), src, np.int8))))
        n_prim_s1 += len(brows) if src else 0
    t = pa.concat_tables(tables)
    d = {c: t[c].to_numpy() for c in predict.CHECKPOINT_COLUMNS + ("src",)}
    log(f"{len(plan):,} blocks: {len(have_p & {k for k, _, _ in plan}):,} from the primary run "
        f"({n_prim_s1:,} S1), the rest from the fallback; {len(t):,} candidate pairs")
    keys = ranker.pair_keys(d["s1_row"], d["target_row"])
    if len(np.unique(keys)) != len(keys):
        raise SystemExit("duplicate (S1, target) pairs")

    n_s1 = len(index.s1_ids)
    order = np.lexsort((d["s1_rank"], d["s1_row"]))
    cand_s1, cand_t = d["s1_row"][order], d["target_row"][order]
    sel = d["p"] >= np.where(d["src"] == 1, thr[1], thr[0]).astype(np.float32)
    ms1, mt, mp = d["s1_row"][sel], d["target_row"][sel], d["p"][sel]
    best = matcher.best_s1_per_target(ms1.astype(np.int64), mt.astype(np.int64), mp)
    msrc = d["src"][sel][best]
    ms1, mt, mp = ms1[best], mt[best], mp[best]
    if len(np.unique(mt)) != len(mt):
        raise SystemExit("a target is matched to more than one S1")
    morder = np.lexsort((mt, -mp, ms1))
    ms1, mt = ms1[morder], mt[morder]
    if not np.isin(ranker.pair_keys(ms1, mt), keys).all():
        raise SystemExit("a match is not among the candidates")
    src = index.pool.source
    prefix = pc.utf8_slice_codeunits(index.pool.entity_id, 0, 3).to_numpy(zero_copy_only=False)
    for arr in (cand_t, mt):
        if not (prefix[arr] == np.where(src[arr] == 2, "S2-", "S3-")).all():
            raise SystemExit("target ID prefix does not match its source")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_id_lists(out_dir / "candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids",
                            index.s1_ids, cand_s1, index.pool.entity_id, cand_t)
    write_id_lists(out_dir / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids",
                            index.s1_ids, ms1, index.pool.entity_id, mt)
    n_match = np.bincount(ms1, minlength=n_s1)
    summary = {"s1_total": int(n_s1), "s1_from_primary": int(n_prim_s1), "thresholds": thr,
               "candidate_pairs": int(len(cand_s1)), "pairs_above_threshold": int(sel.sum()),
               "pairs_removed_by_one_s1_per_target": int(sel.sum() - len(mt)), "matched_pairs": int(len(mt)),
               "matched_from_primary": int((msrc == 1).sum()), "s1_with_matches": int((n_match > 0).sum()),
               "s1_empty": int((n_match == 0).sum()),
               "matched_by_source": {"S2": int((src[mt] == 2).sum()), "S3": int((src[mt] == 3).sum())},
               "outputs": str(out_dir)}
    log(json.dumps(summary))
    if args.validate:
        res = subprocess.run([sys.executable, str(REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"),
                              "--matching", str(out_dir / "matching_results.tsv"),
                              "--candidate", str(out_dir / "candidate_pairs.tsv"),
                              "--test-dir", str(cfg.data_dir / args.split), "--check-ids"], capture_output=True, text=True)
        log(f"validator exit {res.returncode}:\n{res.stdout}{res.stderr}")
        summary["validator_exit"] = res.returncode
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
