"""E03a: bounded experiment for a phonetic-skeleton (Indic) retrieval channel on the dev subset.

Variants (all on ``name_stripped`` skeletons, see ``ber.retrieval.indic``):

* ``I1 indic<=1000``: index only targets whose name is Indic or mixed script, DF cap 1000;
* ``I2 indic<=1000@50``: same, keep the 50 best per S1;
* ``I3 indic<=100``: same index, DF cap 100;
* ``I4 skel_all<=1000``: index every target (control: what a skeleton channel does on Latin);
* ``I5 indic_key`` / ``I6 indic_key<=100``: exact match of the sorted skeleton key of the whole
  name against Indic/mixed targets (uncapped / DF cap 100). Added after the first trial showed
  that Indic names use a tiny vocabulary (~230 distinct skeleton keys), so single-token
  skeleton matching is suppressed by any DF cap.

The key number is the true pairs a variant retrieves that G (name<=10000 | addr<=1000 |
strong name) does not, against the extra candidates it adds. Outputs go to
``work/experiments/retrieval/indic/``. Nothing is adopted automatically.

Run from code/business_entity_resolution/src:  python ../experiments/e03_indic.py [--limit N]
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
from ber import io, tokens  # noqa: E402
from ber.config import Config  # noqa: E402
from ber.retrieval.channels import C1, C2, K, Channel, ChannelRunner  # noqa: E402
from ber.retrieval.evaluate import RetrievalEvaluator, _quantiles  # noqa: E402
from ber.retrieval.indic import skeletons  # noqa: E402
from ber.retrieval.union import union_candidates  # noqa: E402

G_CHANNELS = (C1(10000), C2(1000), K())
VARIANTS = {
    "I1 indic<=1000": ("indic", Channel("indic<=1000", "name_skel", 1000)),
    "I2 indic<=1000@50": ("indic", Channel("indic<=1000@50", "name_skel", 1000, 50)),
    "I3 indic<=100": ("indic", Channel("indic<=100", "name_skel", 100)),
    "I4 skel_all<=1000": ("all", Channel("skel_all<=1000", "name_skel", 1000)),
    # Exact sorted skeleton key of the whole name (the skeleton analogue of the strong name).
    "I5 indic_key": ("indic", Channel("indic_key", "name_skel_key", None)),
    "I6 indic_key<=100": ("indic", Channel("indic_key<=100", "name_skel_key", 100)),
}


def skel_key(skel):
    """Sorted unique skeleton keys of a name, as one exact-match key."""
    return " ".join(sorted(set(skel.split())))


def load_split(cfg):
    return pq.read_table(cfg.work_dir / "cache" / "cv_v1" / "train_s1_split.parquet").to_pandas()


def skeleton_pools(cfg, pool, log):
    """Per index kind ('indic', 'all') and country: a CountryPool holding the ``name_skel`` field."""
    t = tokens.read_normalized(cfg, "train", (2, 3), ["country", "name_script", "name_stripped"])
    country = t["country"].to_numpy(zero_copy_only=False)
    script = script_group(t["name_script"].to_numpy(zero_copy_only=False))
    names = t["name_stripped"].to_pylist()
    del t
    t0 = time.time()
    skel_all = skeletons(names)
    log(f"skeletons of {len(names):,} target names in {time.time() - t0:.0f}s")
    out = {"indic": {}, "all": {}}
    for kind in out:
        for c in pool.countries:
            sel = country == c
            if kind == "indic":
                sel &= np.isin(script, ["indic", "mixed"])
            idx = np.flatnonzero(sel).astype(np.int32)
            cp = tokens.CountryPool(country=c, target_idx=idx)
            cp.fields["name_skel"] = tokens.build_field_index(pa.array([skel_all[i] for i in idx], pa.string()), "tokens")
            cp.fields["name_skel_key"] = tokens.build_field_index(
                pa.array([skel_key(skel_all[i]) for i in idx], pa.string()), "key")
            out[kind][c] = cp
            fi = cp.fields["name_skel"]
            log(f"skeleton index {kind}/{c}: {len(idx):,} targets, {fi.n_tokens:,} keys, {fi.nbytes() / 2**20:,.0f} MB")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args(argv)
    t_start = time.time()
    log = lambda m: print(f"[{time.time() - t_start:7.1f}s | peak {peak_mb():6,.0f} MB] {m}", flush=True)
    cfg = Config.discover()
    out_dir = cfg.work_dir / "experiments" / "retrieval" / "indic"
    out_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(cfg)
    dev_ids = split.loc[split["dev"], "entity_id"].tolist()
    s1 = tokens.read_normalized(cfg, "train", (1,), ["entity_id", "country", "name_fold", "address_norm",
                                                     "name_key", "name_stripped"])
    s1 = s1.filter(pc.is_in(s1["entity_id"], value_set=pa.array(dev_ids)))
    if args.limit:
        s1 = s1.slice(0, args.limit)
    n_q = len(s1)
    q_ids = s1["entity_id"].to_pandas().astype(str)
    q_country = s1["country"].to_numpy(zero_copy_only=False)
    q_skel_list = skeletons(s1["name_stripped"].to_pylist())
    q_skel = pa.array(q_skel_list, pa.string())
    q_skel_key = pa.array([skel_key(x) for x in q_skel_list], pa.string())

    pool = tokens.build_target_pool(cfg, "train", fields=("name_fold", "address_norm", "name_key"), log=log)
    skel_pools = skeleton_pools(cfg, pool, log)

    gt = io.load_ground_truth(cfg).pairs
    gt = gt[gt["source1_entity_id"].isin(set(q_ids))]
    truth_q = pd.Index(q_ids).get_indexer(gt["source1_entity_id"].astype(str))
    truth_t = pd.Index(pool.entity_id.to_pandas().astype(str)).get_indexer(gt["target_entity_id"].astype(str))
    t_script = script_group(pool.name_script.take(pa.array(truth_t)).to_numpy(zero_copy_only=False))
    strata = {
        "all": np.ones(len(truth_q), bool),
        "target Indic": t_script == "indic", "target mixed": t_script == "mixed", "target Latin (control)": t_script == "latin",
        "country=US": q_country[truth_q] == "US", "country=India": q_country[truth_q] == "India",
        "target=S2": pool.source[truth_t] == 2, "target=S3": pool.source[truth_t] == 3,
        "India & target Indic": (q_country[truth_q] == "India") & (t_script == "indic"),
    }
    log(f"{n_q:,} dev queries, {len(truth_q):,} true pairs")

    ev_g = RetrievalEvaluator(n_q, truth_q, truth_t)
    ev_v = {v: RetrievalEvaluator(n_q, truth_q, truth_t) for v in VARIANTS}
    ev_gv = {v: RetrievalEvaluator(n_q, truth_q, truth_t) for v in VARIANTS}
    extra = {v: np.zeros(n_q, dtype=np.int64) for v in VARIANTS}
    seconds = {"G": 0.0, **{v: 0.0 for v in VARIANTS}}
    for c, cpool in pool.countries.items():
        rows = np.flatnonzero(q_country == c)
        if not len(rows):
            continue
        sub = s1.take(pa.array(rows))
        g_runner = ChannelRunner(cpool, {f: tokens.query_matrix(sub[f], tokens.FIELD_MODES[f], cpool.fields[f])
                                         for f in ("name_fold", "address_norm", "name_key")})
        v_runners = {}
        for kind in skel_pools:
            cp = skel_pools[kind][c]
            v_runners[kind] = ChannelRunner(cp, {
                "name_skel": tokens.query_matrix(q_skel.take(pa.array(rows)), "tokens", cp.fields["name_skel"]),
                "name_skel_key": tokens.query_matrix(q_skel_key.take(pa.array(rows)), "key", cp.fields["name_skel_key"]),
            })
        for start in range(0, len(rows), args.chunk):
            stop = min(start + args.chunk, len(rows))
            t0 = time.time()
            g = union_candidates([g_runner.run(ch, start, stop)[:3] for ch in G_CHANNELS], stop - start)
            g_keys = (g["q"].astype(np.int64) << 32) | g["t"]
            ev_g.add(rows[start + g["q"]], g["t"], g["rank"])
            seconds["G"] += time.time() - t0
            for name, (kind, channel) in VARIANTS.items():
                t0 = time.time()
                q, t, score, rank = v_runners[kind].run(channel, start, stop)
                ev_v[name].add(rows[start + q], t, rank)
                keys = (q.astype(np.int64) << 32) | t
                new = ~np.isin(keys, g_keys, assume_unique=True)
                extra[name][rows[start:stop]] += np.bincount(q[new], minlength=stop - start)
                u = union_candidates([(g["q"], g["t"], g["score"]), (q, t, score)], stop - start)
                ev_gv[name].add(rows[start + u["q"]], u["t"], u["rank"])
                seconds[name] += time.time() - t0
        log(f"{c}: {len(rows):,} queries done")

    found_g = ev_g.retrieved()
    report = {"n_queries": n_q, "n_true_pairs": int(len(truth_q)),
              "G": {"summary": ev_g.summary(), "seconds": round(seconds["G"], 1),
                    "recall_by_stratum": {k: round(float(found_g[m].mean()), 4) for k, m in strata.items()}},
              "variants": {}}
    for name in VARIANTS:
        found_v, found_gv = ev_v[name].retrieved(), ev_gv[name].retrieved()
        unique = found_gv & ~found_g
        s_g, s_gv = ev_g.summary(), ev_gv[name].summary()
        report["variants"][name] = {
            "standalone": {k: ev_v[name].summary()[k] for k in ("pair_recall", "candidates")},
            "extra_candidates_beyond_G": _quantiles(extra[name]),
            "unique_true_pairs_beyond_G": int(unique.sum()),
            "unique_share_of_all_true_pairs": round(float(unique.mean()), 5),
            "overlap_with_G_share_of_channel_hits": round(float((found_v & found_g).sum() / max(found_v.sum(), 1)), 4),
            "G_plus_channel": {k: s_gv[k] for k in ("pair_recall", "s1_recall", "oracle_f05", "pct_all_true_retrieved")},
            "delta_vs_G": {k: round(s_gv[k] - s_g[k], 4) for k in ("pair_recall", "s1_recall", "oracle_f05")},
            "by_stratum": {k: {"n_pairs": int(m.sum()), "channel_recall": round(float(found_v[m].mean()), 4),
                               "G_recall": round(float(found_g[m].mean()), 4),
                               "G_plus_recall": round(float(found_gv[m].mean()), 4),
                               "unique_beyond_G": int(unique[m].sum())} for k, m in strata.items()},
            "seconds": round(seconds[name], 1), "ms_per_query": round(1000 * seconds[name] / n_q, 3),
        }
    report["peak_working_set_mb"] = round(peak_mb())
    report["total_seconds"] = round(time.time() - t_start, 1)
    (out_dir / "indic_results.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log(f"wrote {out_dir / 'indic_results.json'}")
    for name, r in report["variants"].items():
        print(f"{name:20s} unique {r['unique_true_pairs_beyond_G']:6d}  extra mean {r['extra_candidates_beyond_G']['mean']:8.1f} "
              f"p95 {r['extra_candidates_beyond_G']['p95']:8.1f}  dRecall {r['delta_vs_G']['pair_recall']:+.4f}  "
              f"dOracleF {r['delta_vs_G']['oracle_f05']:+.4f}  {r['ms_per_query']:.2f} ms/q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
