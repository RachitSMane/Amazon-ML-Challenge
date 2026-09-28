"""E05: stage-2 LightGBM variants on the rows saved by ``e04_matcher.py --save-rows`` (no new candidates).

Same train / early-stopping / dev rows, same threshold tuning and one-S1-per-target decision and
same macro F0.5 as e04, so every variant is compared with e04's model on identical dev S1.
Optionally also scores e04-style saved models (``--score-model``) on these dev rows.

    python ../experiments/e05_stage2_variants.py --rows-from v2_t100000_v20000 --variant lr03_l127:learning_rate=0.03,num_leaves=127
    python ../experiments/e05_stage2_variants.py --rows-from v2_t100000_v20000 --score-model full_t60000_v20000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber import matcher, ranker  # noqa: E402
from ber.config import Config  # noqa: E402


def parse_variant(spec):
    name, _, kv = spec.partition(":")
    params = {}
    for item in filter(None, kv.split(",")):
        k, v = item.split("=")
        params[k] = int(v) if v.lstrip("-").isdigit() else float(v)
    return name, params


def evaluate(p, d):
    thr, best, curve = matcher.tune_threshold(d["qd"], d["td"], p, d["yd"], d["n_dev"], d["n_true"])
    return thr, best, curve


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-from", required=True, help="e04 out-name whose rows.npz to use")
    ap.add_argument("--variant", action="append", default=[], help="name:param=value,...")
    ap.add_argument("--score-model", action="append", default=[], help="e04 out-name of a saved model to score")
    args = ap.parse_args(argv)
    import lightgbm as lgb

    cfg = Config.discover()
    base = cfg.work_dir / "experiments" / "matcher"
    z = np.load(base / args.rows_from / "rows.npz")
    X, y, q, t, role, dev_rows = z["X"], z["y"], z["q"], z["t"], z["role"], z["dev_rows"]
    truth_q = z["truth_q"]
    names = json.loads((base / args.rows_from / "features.json").read_text(encoding="utf-8")) \
        if (base / args.rows_from / "features.json").exists() else None
    local = np.full(int(max(q.max(), dev_rows.max(), truth_q.max())) + 1, -1, dtype=np.int64)
    local[dev_rows] = np.arange(len(dev_rows))
    dev = role == 2
    d = {"qd": local[q[dev]], "td": t[dev], "yd": y[dev], "n_dev": len(dev_rows),
         "n_true": np.bincount(local[truth_q[np.isin(truth_q, dev_rows)]], minlength=len(dev_rows))}
    out = {}
    for m in args.score_model:
        b = lgb.Booster(model_file=str(base / m / "stage2_model.txt"))
        thr, best, _ = evaluate(b.predict(X[dev]), d)
        out[f"model:{m}"] = {"threshold": thr, **best}
        print(json.dumps({f"model:{m}": out[f"model:{m}"]}), flush=True)
    fit, es = role == 0, role == 1
    ones = np.ones(len(y), np.float32)
    for spec in args.variant:
        name, params = parse_variant(spec)
        t0 = time.time()
        b, evals = ranker.train_lgbm(X[fit], y[fit], ones[fit], X[es], y[es], ones[es],
                                     names or [f"f{i}" for i in range(X.shape[1])], params=params)
        thr, best, curve = evaluate(b.predict(X[dev]), d)
        res = {"params": params, "trees": b.best_iteration, "train_s": round(time.time() - t0, 1),
               "threshold": thr, **best}
        out[name] = res
        vdir = base / args.rows_from / "variants" / name
        vdir.mkdir(parents=True, exist_ok=True)
        b.save_model(str(vdir / "stage2_model.txt"))
        (vdir / "dev_metrics.json").write_text(json.dumps({**res, "threshold_curve": curve}, indent=1), encoding="utf-8")
        print(json.dumps({name: res}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
