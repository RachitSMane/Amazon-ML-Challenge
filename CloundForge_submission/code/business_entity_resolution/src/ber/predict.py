"""End-to-end inference: locked stage 1 + stage 2 -> candidate_pairs.tsv, matching_results.tsv.

Uses no labels: the ground-truth loader is disabled for the whole process (``main``).

Per block of ``chunk`` S1 of one country (resumable, one checkpoint file per block):

1. G+Indic retrieval (channels identical to ``experiments/e03_ranker.py``: C1 name<=10000,
   C2 address<=1000, strong-name key, Indic skeleton key), merged per (S1, target).
2. Union-rank cap 500, stage-1 features, ``model_GI_unweighted.txt`` with its first 400
   trees (raw margin), top 50 per S1 (``matcher.stage1_top``).
3. The 62 stage-2 features in ``features.json`` order and the saved stage-2 model's p(match).
4. Checkpoint: ``(s1_row, target_row, s1_rank, s1_score, p)`` to ``<run_dir>/blocks/``.

``finalize`` then reads every checkpoint and, over the whole split at once, writes
``candidate_pairs.tsv`` (the top-50 of every S1, stage-1 order) and ``matching_results.tsv``
(p >= threshold and the pair is the best-scoring S1 of its target), one row per S1 of the
split, empty lists included. IDs are written exactly as in the source files.

Run from code/business_entity_resolution/src:

    python -m ber.predict --split test --run-name full --out ../../../output --validate
    python -m ber.predict --split test --run-name smoke_n2000 --sample 2000 --validate

An interrupted run resumes with the same command (finished blocks are skipped).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import io, matcher, ranker, tokens
from .config import REPO_ROOT, Config
from .retrieval.channels import C1, C2, K, Channel, ChannelRunner
from .retrieval.indic import INDIC_KEY_FIELD, build_indic_key_pools, skeleton_keys

# Identical to experiments/e03_ranker.py (G, GI, QUERY_COLUMNS, TARGET_COLUMNS).
G_CHANNELS = {"name": C1(10000), "addr": C2(1000), "key": K()}
GI_CHANNELS = dict(G_CHANNELS, ikey=Channel("indic_key", INDIC_KEY_FIELD))
QUERY_COLUMNS = ["entity_id", "country", "name_fold", "address_norm", "name_key", "name_stripped", "name_script",
                 "address_missing", "house_numbers", "unit_numbers", "postcode", "source"]
TARGET_COLUMNS = ["country", "name_script", "name_stripped", "address_missing", "house_numbers", "unit_numbers",
                  "postcode", "source"]
MAIN_FIELDS = ("name_fold", "address_norm", "name_key")

DEFAULT_STAGE2_DIR = Path("experiments") / "matcher" / "full_t60000_v20000"  # under the work directory
CHECKPOINT_COLUMNS = ("s1_row", "target_row", "s1_rank", "s1_score", "p")


def _uniq(cols):
    return list(dict.fromkeys(cols))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _no_ground_truth(*_, **__):
    raise RuntimeError("ground truth must not be loaded during inference")


def disable_ground_truth():
    """Make every ground-truth loader in ``ber.io`` raise (inference must never see labels)."""
    for name in ("load_ground_truth", "read_ground_truth_tsv", "_write_ground_truth_cache"):
        setattr(io, name, _no_ground_truth)


class Models:
    """The locked stage-1 ranker and the saved stage-2 matcher with its configuration."""

    def __init__(self, cfg, stage2_dir=None):
        import lightgbm as lgb

        self.stage2_dir = Path(stage2_dir) if stage2_dir else cfg.work_dir / DEFAULT_STAGE2_DIR
        self.config = json.loads((self.stage2_dir / "config.json").read_text(encoding="utf-8"))
        self.features = json.loads((self.stage2_dir / "features.json").read_text(encoding="utf-8"))
        self.stage1_cfg = self.config["stage1"]
        if self.stage1_cfg != matcher.STAGE1:
            raise ValueError(f"stage-1 configuration differs: saved {self.stage1_cfg} vs code {matcher.STAGE1}")
        self.threshold = float(self.config["threshold"])
        self.stage1_path = cfg.work_dir / "experiments" / "ranker" / self.stage1_cfg["model_file"]
        self.stage2_path = self.stage2_dir / "stage2_model.txt"
        self.stage1_sha, self.stage2_sha = sha256(self.stage1_path), sha256(self.stage2_path)
        if self.stage1_sha != self.config["stage1_model_sha256"]:
            raise ValueError("stage-1 model file does not match the hash recorded at stage-2 training")
        self.stage1 = lgb.Booster(model_file=str(self.stage1_path))
        self.stage2 = lgb.Booster(model_file=str(self.stage2_path))
        self.stage1_names = ranker.feature_names(tuple(GI_CHANNELS))
        if self.stage1.feature_name() != self.stage1_names:
            raise ValueError("stage-1 model features differ from ranker.feature_names(G+Indic)")
        if self.stage1.num_trees() < self.stage1_cfg["trees"]:
            raise ValueError("stage-1 model has fewer trees than configured")
        expected = matcher.stage2_feature_names(self.stage1_names)
        if self.features != expected or self.stage2.feature_name() != expected:
            raise ValueError("stage-2 features.json / model features differ from matcher.stage2_feature_names")

    def fingerprint(self):
        return {"stage1": self.stage1_cfg, "stage1_sha256": self.stage1_sha, "stage2_sha256": self.stage2_sha,
                "threshold": self.threshold, "features": self.features}


class Index:
    """Queries (every S1 of the split), the indexed S2+S3 pool and per-record arrays. No labels."""

    def __init__(self, cfg, split, log=print):
        self.split, self.log = split, log
        t0 = time.time()
        s1 = tokens.read_normalized(cfg, split, (1,), _uniq(QUERY_COLUMNS + matcher.STRING_COLUMNS))
        self.s1 = s1
        self.s1_ids = s1["entity_id"].combine_chunks()
        self.q_country = s1["country"].to_numpy(zero_copy_only=False)
        self.queries = ranker.RecordArrays.from_table(s1)
        self.q_ikey = pa.array(skeleton_keys(s1["name_stripped"].to_pylist()), pa.string())
        self.qstr = {c: s1[c].combine_chunks() for c in matcher.STRING_COLUMNS}
        log(f"{split}: {len(s1):,} S1 loaded")

        self.pool = tokens.build_target_pool(cfg, split, fields=MAIN_FIELDS)
        tt = tokens.read_normalized(cfg, split, (2, 3), _uniq(["entity_id"] + TARGET_COLUMNS + matcher.STRING_COLUMNS))
        if not pc.all(pc.equal(tt["entity_id"], self.pool.entity_id)).as_py():
            raise ValueError("target table order differs from the target pool")
        self.targets = ranker.RecordArrays.from_table(tt)
        self.ikey_pools = build_indic_key_pools(tt["name_stripped"].to_pylist(),
                                                tt["country"].to_numpy(zero_copy_only=False),
                                                tt["name_script"].to_numpy(zero_copy_only=False),
                                                list(self.pool.countries))
        self.tstr = {c: tt[c].combine_chunks() for c in matcher.STRING_COLUMNS}
        del tt
        ranker.add_target_statistics(self.targets, self.pool, GI_CHANNELS, extra_pools={"ikey": self.ikey_pools})
        self.index_seconds = time.time() - t0
        log(f"{split}: {len(self.pool.entity_id):,} targets indexed ({', '.join(self.pool.countries)}) "
            f"in {self.index_seconds:.0f}s")

    def block_plan(self, rows, chunk):
        """[(key, country, rows of the block)] in a fixed order: per pool country, ``chunk`` rows."""
        rows = np.asarray(rows)
        plan = []
        for c in self.pool.countries:
            crow = rows[self.q_country[rows] == c]
            for start in range(0, len(crow), chunk):
                plan.append((f"{c}_{start:08d}", c, crow[start:start + chunk]))
        return plan

    def unplanned(self, rows):
        """S1 rows whose country has no targets (they get no candidates)."""
        rows = np.asarray(rows)
        return rows[~np.isin(self.q_country[rows], list(self.pool.countries))]

    def merged_blocks(self, plan, skip=()):
        """Yield (key, block rows, merged G+Indic candidates, query stats) for the planned blocks."""
        by_country = {}
        for key, c, brows in plan:
            by_country.setdefault(c, []).append((key, brows))
        for c, blocks in by_country.items():
            todo = [(k, b) for k, b in blocks if k not in skip]
            if not todo:
                continue
            crow = np.concatenate([b for _, b in blocks])
            cpool = self.pool.countries[c]
            sub = self.s1.take(pa.array(crow))
            main = ChannelRunner(cpool, {f: tokens.query_matrix(sub[f], tokens.FIELD_MODES[f], cpool.fields[f])
                                         for f in MAIN_FIELDS})
            icp = self.ikey_pools[c]
            irun = ChannelRunner(icp, {INDIC_KEY_FIELD: tokens.query_matrix(self.q_ikey.take(pa.array(crow)), "key",
                                                                             icp.fields[INDIC_KEY_FIELD])})
            runners = {"name": main, "addr": main, "key": main, "ikey": irun}
            offset = 0
            for key, brows in blocks:
                start, stop = offset, offset + len(brows)
                offset = stop
                if key in skip:
                    continue
                parts = {p: runners[p].run(ch, start, stop) for p, ch in GI_CHANNELS.items()}
                m = ranker.merge_channels([parts[p] for p in GI_CHANNELS], stop - start)
                qstats = ranker.query_block_stats(main, runners, GI_CHANNELS, start, stop)
                yield key, brows, m, qstats


def score_block(index, models, brows, m, qstats, threads=None):
    """Stage 1 + stage 2 on one merged block. Returns a dict of per-pair arrays (and features)."""
    st1 = models.stage1_cfg
    n = len(brows)
    mc, _ = matcher.cap_merged(m, st1["cap"])
    X1 = ranker.pair_features(mc, tuple(GI_CHANNELS), qstats, brows, index.queries, index.targets)
    top, raw, rank = matcher.stage1_top(mc.q, mc.t, X1, models.stage1, n, trees=st1["trees"], top_k=st1["top_k"],
                                        threads=threads)
    q_blk, t = mc.q[top], mc.t[top]
    s1_row = brows[q_blk]
    qstr = {c: a.take(pa.array(s1_row)) for c, a in index.qstr.items()}
    tstr = {c: a.take(pa.array(t)) for c, a in index.tstr.items()}
    X = matcher.stage2_features(X1[top], models.stage1_names, raw, rank, q_blk, n, qstr, tstr)
    p = models.stage2.predict(X, num_threads=threads or 0) if len(X) else np.zeros(0)
    return {"s1_row": s1_row.astype(np.int32), "target_row": t.astype(np.int32), "s1_rank": rank.astype(np.int16),
            "s1_score": raw, "p": p.astype(np.float32), "X": X}


# --------------------------------------------------------------------------- run + finalize

def run_blocks(index, models, rows, run_dir, chunk=500, threads=None, log=print):
    """Score every planned block not yet checkpointed in ``run_dir/blocks``."""
    blocks_dir = run_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    plan = index.block_plan(rows, chunk)
    done = {p.stem for p in blocks_dir.glob("*.parquet")}
    todo = [k for k, _, _ in plan if k not in done]
    n_rows = sum(len(b) for _, _, b in plan)
    n_todo = sum(len(b) for k, _, b in plan if k not in done)
    log(f"{len(plan):,} blocks ({n_rows:,} S1); {len(plan) - len(todo):,} already checkpointed, "
        f"{len(todo):,} to score ({n_todo:,} S1)")
    t_pass, processed = time.time(), 0
    for key, brows, m, qstats in index.merged_blocks(plan, skip=done):
        out = score_block(index, models, brows, m, qstats, threads)
        table = pa.table({c: out[c] for c in CHECKPOINT_COLUMNS})
        tmp = blocks_dir / f"{key}.parquet.tmp"
        pq.write_table(table, tmp)
        os.replace(tmp, blocks_dir / f"{key}.parquet")
        processed += len(brows)
        el = time.time() - t_pass
        log(f"block {key}: {len(brows)} S1, {len(out['p']):,} pairs | {processed:,}/{n_todo:,} S1 "
            f"({100 * processed / max(n_todo, 1):.1f}%), elapsed {el:.0f}s, ETA {el / processed * (n_todo - processed):.0f}s")
    return plan


def _write_id_lists(path, header, s1_ids, s1_row, target_ids, target_row):
    """One line per S1 (in ``s1_ids`` order): ``id<TAB>comma-joined target ids`` (pairs pre-sorted)."""
    n = len(s1_ids)
    counts = np.bincount(s1_row, minlength=n)
    offsets = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    lists = pa.ListArray.from_arrays(pa.array(offsets), target_ids.take(pa.array(target_row)))
    lines = pc.binary_join_element_wise(s1_ids, pc.binary_join(lists, ","), "\t")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for start in range(0, n, 200_000):
            f.write("\n".join(lines.slice(start, 200_000).to_pylist()) + "\n")


def finalize(index, models, plan, run_dir, out_dir, log=print):
    """Global decision over all checkpoints and the two submission files (every S1 of the split)."""
    expected = {k for k, _, _ in plan}
    files = {p.stem: p for p in (run_dir / "blocks").glob("*.parquet")}
    missing = expected - set(files)
    if missing:
        raise RuntimeError(f"{len(missing)} blocks have no checkpoint yet (e.g. {sorted(missing)[:3]})")
    t = pa.concat_tables([pq.read_table(files[k]) for k in sorted(expected)])
    d = {c: t[c].to_numpy() for c in CHECKPOINT_COLUMNS}
    keys = ranker.pair_keys(d["s1_row"], d["target_row"])
    if len(np.unique(keys)) != len(keys):
        raise RuntimeError("duplicate (S1, target) pairs in the checkpoints")
    planned_rows = np.concatenate([b for _, _, b in plan]) if plan else np.zeros(0, int)
    if not np.isin(d["s1_row"], planned_rows).all():
        raise RuntimeError("checkpoint pairs for S1 outside the plan")

    n_s1 = len(index.s1_ids)
    order = np.lexsort((d["s1_rank"], d["s1_row"]))
    cand_s1, cand_t = d["s1_row"][order], d["target_row"][order]

    sel = d["p"] >= models.threshold
    ms1, mt, mp = d["s1_row"][sel], d["target_row"][sel], d["p"][sel]
    best = matcher.best_s1_per_target(ms1.astype(np.int64), mt.astype(np.int64), mp)
    ms1, mt, mp = ms1[best], mt[best], mp[best]
    if len(np.unique(mt)) != len(mt):
        raise RuntimeError("a target is matched to more than one S1")
    morder = np.lexsort((mt, -mp, ms1))
    ms1, mt = ms1[morder], mt[morder]

    src = index.pool.source
    prefix = pc.utf8_slice_codeunits(index.pool.entity_id, 0, 3).to_numpy(zero_copy_only=False)
    for arr in (cand_t, mt):
        exp = np.where(src[arr] == 2, "S2-", "S3-")
        if not (prefix[arr] == exp).all():
            raise RuntimeError("target ID prefix does not match its source")

    out_dir.mkdir(parents=True, exist_ok=True)
    cand_path, match_path = out_dir / "candidate_pairs.tsv", out_dir / "matching_results.tsv"
    _write_id_lists(cand_path, "source1_entity_id\tcandidate_entity_ids", index.s1_ids, cand_s1,
                    index.pool.entity_id, cand_t)
    _write_id_lists(match_path, "source1_entity_id\tmatched_entity_ids", index.s1_ids, ms1,
                    index.pool.entity_id, mt)

    n_cand = np.bincount(cand_s1, minlength=n_s1)
    n_match = np.bincount(ms1, minlength=n_s1)
    in_plan = np.zeros(n_s1, bool)
    in_plan[planned_rows] = True
    summary = {
        "split": index.split, "s1_total": int(n_s1), "s1_scored": int(in_plan.sum()),
        "candidate_pairs": int(len(cand_s1)), "pairs_above_threshold": int(sel.sum()),
        "pairs_removed_by_one_s1_per_target": int(sel.sum() - len(mt)), "matched_pairs": int(len(mt)),
        "scored_s1_with_matches": int((n_match[in_plan] > 0).sum()),
        "scored_s1_empty": int((n_match[in_plan] == 0).sum()),
        "scored_s1_multi_match": int((n_match[in_plan] > 1).sum()),
        "max_matches_per_s1": int(n_match.max()) if n_s1 else 0,
        "scored_s1_without_candidates": int((n_cand[in_plan] == 0).sum()),
        "candidates_per_scored_s1": round(float(n_cand[in_plan].mean()), 2) if in_plan.any() else 0.0,
        "matched_by_source": {"S2": int((src[mt] == 2).sum()), "S3": int((src[mt] == 3).sum())},
        "matched_by_country": {c: int((index.q_country[ms1] == c).sum()) for c in np.unique(index.q_country)},
        "outputs": {"candidate_pairs": str(cand_path), "matching_results": str(match_path)},
    }
    log(f"wrote {cand_path} and {match_path}: {json.dumps(summary)}")
    return summary


def select_rows(index, sample=None, seed="ber-predict-sample"):
    """All S1 rows, or the ``sample`` rows with the smallest keyed hash of their ID."""
    rows = np.arange(len(index.s1_ids))
    if sample and sample < len(rows):
        from .cv import stable_hash

        h = stable_hash(index.s1_ids.to_pylist(), seed)
        rows = np.sort(np.argsort(h, kind="stable")[:sample])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--run-name", required=True, help="checkpoint folder: <work>/inference/<split>/<run-name>")
    ap.add_argument("--out", help="output folder for the two TSV files (default: <run folder>/output)")
    ap.add_argument("--sample", type=int, help="score only this many S1 (deterministic); others get empty rows")
    ap.add_argument("--stage2-dir", help="saved stage-2 run (default: <work>/experiments/matcher/full_t60000_v20000)")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--validate", action="store_true", help="run student_resource/utils/validate_submission.py")
    args = ap.parse_args(argv)
    disable_ground_truth()
    t_start = time.time()
    log =lambda m: print(f"[{time.time() - t_start:7.1f}s] {m}", flush=True)
    cfg = Config.discover()
    run_dir = cfg.work_dir / "inference" / args.split / args.run_name
    out_dir = Path(args.out).resolve() if args.out else run_dir / "output"
    models = Models(cfg, args.stage2_dir)
    fp = dict(models.fingerprint(), split=args.split, sample=args.sample, chunk=args.chunk)
    run_dir.mkdir(parents=True, exist_ok=True)
    fp_path = run_dir / "run_config.json"
    if fp_path.exists():
        if json.loads(fp_path.read_text(encoding="utf-8")) != json.loads(json.dumps(fp)):
            raise SystemExit(f"{fp_path} was written with a different configuration; use another --run-name")
    else:
        fp_path.write_text(json.dumps(fp, indent=1), encoding="utf-8")
    log(f"stage 1 {models.stage1_cfg}, threshold {models.threshold}, {len(models.features)} stage-2 features; "
        f"checkpoints {run_dir}")

    index = Index(cfg, args.split, log)
    rows = select_rows(index, args.sample)
    skipped = index.unplanned(rows)
    if len(skipped):
        log(f"warning: {len(skipped):,} S1 have a country without targets; they get empty rows")
    plan = run_blocks(index, models, rows, run_dir, args.chunk, args.threads, log)
    summary = finalize(index, models, plan, run_dir, out_dir, log)
    summary.update(total_s=round(time.time() - t_start, 1), index_build_s=round(index.index_seconds, 1))
    if args.validate:
        test_dir = cfg.data_dir / args.split
        res = subprocess.run([sys.executable, str(REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"),
                              "--matching", str(out_dir / "matching_results.tsv"),
                              "--candidate", str(out_dir / "candidate_pairs.tsv"),
                              "--test-dir", str(test_dir), "--check-ids"], capture_output=True, text=True)
        log(f"validator exit {res.returncode}:\n{res.stdout}{res.stderr}")
        summary["validator_exit"] = res.returncode
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
