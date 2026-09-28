"""E06: paired bootstrap of the per-S1 F0.5 difference between two saved stage-2 models on the same dev rows.

    python ../experiments/e06_paired_bootstrap.py --rows-from v2_t100000_v20000 --a full_t60000_v20000 --b v2_t100000_v20000
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber import matcher  # noqa: E402
from ber.config import Config  # noqa: E402


def per_s1(q, t, y, p, thr, n, n_true):
    sel = matcher.decide(q, t, p, thr)
    n_pred = np.bincount(q[sel], minlength=n).astype(float)
    n_hit = np.bincount(q[sel], weights=y[sel], minlength=n)
    b2 = 0.25
    with np.errstate(divide="ignore", invalid="ignore"):
        pr, rc = n_hit / n_pred, n_hit / n_true
        f = (1 + b2) * pr * rc / (b2 * pr + rc)
    f = np.where((n_pred == 0) | (n_hit == 0), 0.0, f)
    return np.where(n_true == 0, (n_pred == 0).astype(float), f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-from", required=True); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    import lightgbm as lgb
    base = Config.discover().work_dir / "experiments" / "matcher"
    z = np.load(base / args.rows_from / "rows.npz")
    dev = z["role"] == 2
    X, y, q, t, dev_rows, truth_q = z["X"][dev], z["y"][dev], z["q"][dev], z["t"][dev], z["dev_rows"], z["truth_q"]
    local = np.full(int(max(z["q"].max(), truth_q.max())) + 1, -1, np.int64); local[dev_rows] = np.arange(len(dev_rows))
    q = local[q]; n = len(dev_rows)
    n_true = np.bincount(local[truth_q[np.isin(truth_q, dev_rows)]], minlength=n).astype(float)
    f = {}
    for name in (args.a, args.b):
        thr = json.loads((base / name / "dev_metrics.json").read_text())["threshold"]
        p = lgb.Booster(model_file=str(base / name / "stage2_model.txt")).predict(X)
        f[name] = per_s1(q, t, y, p, thr, n, n_true)
        print(f"{name}: threshold {thr}, macro F0.5 {f[name].mean():.5f}")
    d = f[args.b] - f[args.a]
    rng = np.random.default_rng(0)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(args.n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(json.dumps({"mean_diff": round(float(d.mean()), 5), "ci95": [round(float(lo), 5), round(float(hi), 5)],
                      "p_diff_le_0": round(float((boots <= 0).mean()), 4), "s1_better": int((d > 0).sum()),
                      "s1_worse": int((d < 0).sum()), "s1_equal": int((d == 0).sum())}))


main()
