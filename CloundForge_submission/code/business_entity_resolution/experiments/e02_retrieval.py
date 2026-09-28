"""E02: retrieval experiments on the 100k-S1 development subset (train pool, read-only).

For every channel and union strategy: candidate-count distribution, pair recall, S1 recall,
oracle F0.5, recall by stratum (country, S1 script, match bucket, target source, target
script, missing address, Indic categories), union-level top-K curves, runtime and memory.
Nothing is written outside ``work/`` and no candidate list is materialized beyond one chunk.

Run from code/business_entity_resolution/src:

    python ../experiments/e02_retrieval.py [--limit N] [--chunk 2000] [--out DIR]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber import cv, io, tokens  # noqa: E402
from ber.config import EXPECTED_ROWS, Config  # noqa: E402
from ber.normalize.text import INDIC_SCRIPTS  # noqa: E402
from ber.retrieval.channels import C1, C2, C3, K, ChannelRunner  # noqa: E402
from ber.retrieval.evaluate import RetrievalEvaluator  # noqa: E402
from ber.retrieval.union import Strategy, union_candidates  # noqa: E402

K_VALUES = (5, 10, 20, 50, 100, 200)
INDIC = {label for _, _, label in INDIC_SCRIPTS}
QUERY_COLUMNS = ["entity_id", "country", "name_script"] + list(tokens.FIELD_MODES)


def peak_mb():
    """Peak working set of this process in MB (Windows), else ru_maxrss."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t)] + \
                       [(f"f{i}", ctypes.c_size_t) for i in range(6)]
        k32 = ctypes.WinDLL("kernel32"); k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2**20
    except (OSError, AttributeError):
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def script_group(scripts):
    s = np.asarray(scripts, dtype=object)
    return np.where(s == "latin", "latin", np.where(np.isin(s, list(INDIC)), "indic", np.where(s == "mixed", "mixed", "other")))


def channels_and_strategies():
    ch = {c.name: c for c in (C1(1000), C1(10000), C2(100), C2(1000), C3(1000), C3(10000), K())}
    s = [Strategy(f"channel {name}", (c,)) for name, c in ch.items()]
    s += [
        Strategy("A strong name", (ch["strong_name"],)),
        Strategy("B name<=1000", (ch["name_fold<=1000"],)),
        Strategy("C name<=1000 | addr<=1000", (ch["name_fold<=1000"], ch["address<=1000"])),
        Strategy("D name<=1000 | addr<=100", (ch["name_fold<=1000"], ch["address<=100"])),
        Strategy("E name<=1000 | addr<=100 | strong", (ch["name_fold<=1000"], ch["address<=100"], ch["strong_name"])),
        Strategy("F name<=10000 | addr<=1000", (ch["name_fold<=10000"], ch["address<=1000"])),
        Strategy("E3 stripped<=1000 | addr<=100 | strong", (ch["name_stripped<=1000"], ch["address<=100"], ch["strong_name"])),
        Strategy("F3 stripped<=10000 | addr<=1000", (ch["name_stripped<=10000"], ch["address<=1000"])),
        Strategy("G name<=10000 | addr<=1000 | strong", (ch["name_fold<=10000"], ch["address<=1000"], ch["strong_name"])),
    ]
    return ch, s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, help="use only the first N dev queries (timing trial)")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--out", help="output folder (default work/experiments/retrieval)")
    args = ap.parse_args(argv)
    t_start = time.time()
    cfg = Config.discover()
    out_dir = Path(args.out) if args.out else cfg.work_dir / "experiments" / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[{time.time() - t_start:7.1f}s | peak {peak_mb():6,.0f} MB] {m}", flush=True)

    # ---- development split
    split = cv.build_train_split(cfg)
    split_path = cv.save_split(cfg, split)
    dev_ids = split.loc[split["dev"], "entity_id"].to_numpy()
    log(f"split saved to {split_path}: {len(split):,} S1, dev {len(dev_ids):,}; "
        f"fold sizes {np.bincount(split['fold']).tolist()}")
    dev_strata = split.loc[split["dev"], "stratum"].value_counts().sort_index().to_dict()

    # ---- queries (dev S1, normalized)
    s1 = tokens.read_normalized(cfg, "train", (1,), QUERY_COLUMNS)
    s1 = s1.filter(pc.is_in(s1["entity_id"], value_set=pa.array(dev_ids.tolist())))
    if args.limit:
        s1 = s1.slice(0, args.limit)
    n_q = len(s1)
    q_ids = s1["entity_id"].to_pandas().astype(str)
    q_country = s1["country"].to_numpy(zero_copy_only=False)
    q_script = script_group(s1["name_script"].to_numpy(zero_copy_only=False))
    bucket = split.set_index("entity_id").loc[q_ids, "bucket"].to_numpy()
    log(f"{n_q:,} dev queries loaded")

    # ---- target pool
    t0 = time.time()
    pool = tokens.build_target_pool(cfg, "train", log=log)
    pool_seconds = time.time() - t0
    assert len(pool) == EXPECTED_ROWS[("train", 2)] + EXPECTED_ROWS[("train", 3)]
    df_report = {f"{c}/{f}": tokens.df_summary(fi) for c, cp in pool.countries.items() for f, fi in cp.fields.items()}
    (out_dir / "df_summary.json").write_text(json.dumps(df_report, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"target pool built in {pool_seconds:.0f}s, index {pool.nbytes() / 2**20:,.0f} MB")

    # ---- ground truth of the dev queries
    gt = io.load_ground_truth(cfg).pairs
    gt = gt[gt["source1_entity_id"].isin(set(q_ids))]
    truth_q = pd.Index(q_ids).get_indexer(gt["source1_entity_id"].astype(str))
    truth_t = pd.Index(pool.entity_id.to_pandas().astype(str)).get_indexer(gt["target_entity_id"].astype(str))
    assert (truth_q >= 0).all() and (truth_t >= 0).all()
    t_script = script_group(pool.name_script.take(pa.array(truth_t)).to_numpy(zero_copy_only=False))
    p_s1_script = q_script[truth_q]
    pair_strata = {
        "country=US": q_country[truth_q] == "US", "country=India": q_country[truth_q] == "India",
        "target=S2": pool.source[truth_t] == 2, "target=S3": pool.source[truth_t] == 3,
        "target address missing": pool.address_missing[truth_t],
        "target address present": ~pool.address_missing[truth_t],
        **{f"bucket={b}": bucket[truth_q] == b for b in cv.BUCKETS[1:]},
        **{f"target script={g}": t_script == g for g in ("latin", "indic", "mixed", "other")},
        "INDIC: S1 name Indic": p_s1_script == "indic",
        "INDIC: target name Indic": t_script == "indic",
        "INDIC: S1 Latin, target Indic": (p_s1_script == "latin") & (t_script == "indic"),
        "INDIC: both Latin": (p_s1_script == "latin") & (t_script == "latin"),
        "INDIC: target mixed script": t_script == "mixed",
    }
    query_strata = {
        "country=US": q_country == "US", "country=India": q_country == "India",
        **{f"bucket={b}": bucket == b for b in cv.BUCKETS},
        **{f"S1 script={g}": q_script == g for g in ("latin", "indic", "mixed", "other")},
    }
    log(f"{len(gt):,} true pairs for the dev queries")

    # ---- retrieval
    channels, strategies = channels_and_strategies()
    evaluators = {s.name: RetrievalEvaluator(n_q, truth_q, truth_t) for s in strategies}
    ch_seconds = {name: 0.0 for name in channels}
    union_seconds = {s.name: 0.0 for s in strategies}
    max_chunk_pairs = 0
    for country, cpool in pool.countries.items():
        rows = np.flatnonzero(q_country == country)
        if not len(rows):
            continue
        sub = s1.take(pa.array(rows))
        qm = {f: tokens.query_matrix(sub[f], tokens.FIELD_MODES[f], cpool.fields[f]) for f in tokens.FIELD_MODES}
        runner = ChannelRunner(cpool, qm)
        for start in range(0, len(rows), args.chunk):
            stop = min(start + args.chunk, len(rows))
            outs = {}
            for name, channel in channels.items():
                t0 = time.time()
                q, t, score, _ = runner.run(channel, start, stop)
                outs[name] = (q, t, score)
                ch_seconds[name] += time.time() - t0
            for s in strategies:
                t0 = time.time()
                u = union_candidates([outs[c.name] for c in s.channels], stop - start)
                max_chunk_pairs = max(max_chunk_pairs, len(u["q"]))
                evaluators[s.name].add(rows[start + u["q"]], u["t"], u["rank"])
                union_seconds[s.name] += time.time() - t0
            if (start // args.chunk) % 10 == 0:
                log(f"{country}: {stop:,}/{len(rows):,} queries")
    retrieval_seconds = sum(ch_seconds.values()) + sum(union_seconds.values())
    log(f"retrieval done in {retrieval_seconds:.0f}s (largest chunk union: {max_chunk_pairs:,} pairs)")

    # ---- reports
    results, stats, strata_out = {}, {}, {}
    n_test = EXPECTED_ROWS[("test", 1)]
    for s in strategies:
        ev = evaluators[s.name]
        full = ev.summary(query_strata=query_strata, pair_strata=pair_strata)
        seconds = sum(ch_seconds[c.name] for c in s.channels) + union_seconds[s.name]
        topk = {}
        for k in K_VALUES:
            sk = ev.summary(k=k)
            topk[k] = {key: sk[key] for key in ("pair_recall", "s1_recall", "oracle_f05", "pct_all_true_retrieved")}
            topk[k]["mean_candidates"] = sk["candidates"]["mean"]
        results[s.name] = {
            "channels": [c.name for c in s.channels],
            **{k: full[k] for k in ("n_queries", "n_true_pairs", "pair_recall", "s1_recall", "oracle_f05",
                                    "pct_zero_candidates", "pct_all_true_retrieved", "pct_any_true_retrieved")},
            "mean_candidates": full["candidates"]["mean"],
            "seconds": round(seconds, 1),
            "ms_per_query": round(1000 * seconds / n_q, 3),
            "est_test_minutes_single_process": round(seconds / n_q * n_test / 60, 1),
            "est_test_pairs_millions": round(full["candidates"]["mean"] * n_test / 1e6, 1),
            "union_topk": topk,
        }
        stats[s.name] = full["candidates"]
        strata_out[s.name] = {"by_query_stratum": full["by_query_stratum"], "by_pair_stratum": full["by_pair_stratum"]}

    meta = {
        "n_dev_queries": n_q, "dev_strata": dev_strata, "split_file": str(split_path),
        "pool_targets": len(pool), "pool_build_seconds": round(pool_seconds, 1),
        "pool_index_mb": round(pool.nbytes() / 2**20, 1),
        "channel_seconds": {k: round(v, 1) for k, v in ch_seconds.items()},
        "union_seconds": {k: round(v, 1) for k, v in union_seconds.items()},
        "retrieval_seconds": round(retrieval_seconds, 1), "chunk_rows": args.chunk,
        "max_chunk_union_pairs": int(max_chunk_pairs),
        "peak_working_set_mb": round(peak_mb(), 0), "total_seconds": round(time.time() - t_start, 1),
    }
    for name, obj in (("strategy_results", results), ("candidate_stats", stats),
                      ("recall_by_stratum", strata_out), ("run_meta", meta)):
        (out_dir / f"{name}.json").write_text(json.dumps(obj, indent=1), encoding="utf-8")
    log(f"wrote results to {out_dir}")
    for name, r in results.items():
        print(f"{name:42s} mean {r['mean_candidates']:9.1f}  recall {r['pair_recall']:.4f}  "
              f"S1 {r['s1_recall']:.4f}  oracleF {r['oracle_f05']:.4f}  {r['ms_per_query']:.2f} ms/q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
