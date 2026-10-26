"""Pre-registered analysis of the held-out test of the lean ladder (PREREG_lean.md): the lean and
lguard ladders on the same noisy models (one tag directory written by evaluate.py).

For each cell (spec, sigma) with both <cell>_lean.jsonl and <cell>_lguard.jsonl:
  * checks that both files hold the same seeds and the same truth per seed, and that each exact
    rung is scored at the truth;
  * worst fair ratio of each ladder (summarize.worst_ratios);
  * a paired percentile bootstrap (the same resampled noisy models for both ladders; 10,000
    replicates, seed 0): 90% intervals, one-sided p-values p = P*(worst ratio >= 1) per ladder,
    and the paired log ratio log(lean / lguard) with p = P*(lean >= lguard).
Three families, each with a Holm correction at level 0.05 over its 21 cells:
  primary   - lean beats sampling at every budget (worst fair ratio < 1);
  secondary - lguard beats sampling (a replication of the README's earlier test, on new seeds);
  secondary - lean has a smaller worst fair ratio than lguard.

Usage: python analyze_test.py results/test7000 [n_boot]   (also writes <dir>/analysis.json)
"""

import glob
import json
import os
import sys

import numpy as np

from summarize import load, worst_ratios


def holm(p, alpha=0.05):
    """Holm's step-down procedure: which of the hypotheses with p-values p are rejected."""
    p = np.asarray(p, float)
    rej = np.zeros(len(p), bool)
    for k, i in enumerate(np.argsort(p)):
        if p[i] > alpha / (len(p) - k):
            break
        rej[i] = True
    return rej


def cell_key(c):  # "M_8_3_0.005" -> (8, 0.005)
    parts = c.split("_")
    return int(parts[1]), float(parts[3])


def main(d, n_boot=10000):
    cells = sorted({os.path.basename(f).rsplit("_", 1)[0] for f in glob.glob(f"{d}/*.jsonl")}, key=cell_key)
    rows = []
    for c in cells:
        fa, fb = f"{d}/{c}_lean.jsonl", f"{d}/{c}_lguard.jsonl"
        if not (os.path.exists(fa) and os.path.exists(fb)):
            print(f"{c}: missing one ladder, skipped")
            continue
        A, B = load(fa), load(fb)
        seeds_a = sorted(json.loads(l)["seed"] for l in open(fa))
        seeds_b = sorted(json.loads(l)["seed"] for l in open(fb))
        assert seeds_a == seeds_b and len(set(seeds_a)) == len(seeds_a), f"{c}: seed sets differ"
        assert np.array_equal(A[1], B[1]), f"{c}: truth differs"
        for names, truth, est, _ in (A, B):
            assert np.array_equal(est[:, names.index("exact")], truth), f"{c}: exact rung is not the truth"
        K = len(A[1])
        wa, wb = worst_ratios(*A), worst_ratios(*B)
        rng = np.random.default_rng(0)
        bs = np.empty((n_boot, 2))
        for i in range(n_boot):
            idx = rng.integers(0, K, K)
            bs[i] = worst_ratios(*A, idx)[0], worst_ratios(*B, idx)[0]
        lr = np.log(bs[:, 0] / bs[:, 1])
        rows.append(dict(cell=c, K=K, mean_acc=float(A[1].mean()),
                         lean=wa[0], lean_at=wa[2], lean_ci=np.quantile(bs[:, 0], [0.05, 0.95]).tolist(),
                         p_lean=float((bs[:, 0] >= 1).mean()),
                         lguard=wb[0], lguard_at=wb[2], lguard_ci=np.quantile(bs[:, 1], [0.05, 0.95]).tolist(),
                         p_lguard=float((bs[:, 1] >= 1).mean()),
                         log_ratio=float(np.log(wa[0] / wb[0])), log_ratio_ci=np.quantile(lr, [0.05, 0.95]).tolist(),
                         p_lean_not_better=float((lr >= 0).mean())))
    if not rows:
        return
    for key, fam in (("p_lean", "holm_lean"), ("p_lguard", "holm_lguard"), ("p_lean_not_better", "holm_better")):
        for r, rej in zip(rows, holm([r[key] for r in rows])):
            r[fam] = bool(rej)
    print(f"{len(rows)} cells, {n_boot} paired bootstrap replicates, Holm at 0.05 within each family")
    print(f"{'cell':13s} {'K':>3s} {'acc':>6s} | {'lean':>6s} {'90% CI':>15s} {'p':>6s} H | "
          f"{'lguard':>6s} {'90% CI':>15s} {'p':>6s} H | {'lean/lguard':>11s} {'90% CI':>15s} {'p':>6s} H | lean binding")
    for r in rows:
        print(f"{r['cell']:13s} {r['K']:3d} {r['mean_acc']:6.4f} | {r['lean']:6.3f} [{r['lean_ci'][0]:6.3f},{r['lean_ci'][1]:6.3f}] "
              f"{r['p_lean']:6.4f} {'*' if r['holm_lean'] else ' '} | {r['lguard']:6.3f} "
              f"[{r['lguard_ci'][0]:6.3f},{r['lguard_ci'][1]:6.3f}] {r['p_lguard']:6.4f} {'*' if r['holm_lguard'] else ' '} | "
              f"{np.exp(r['log_ratio']):11.3f} [{np.exp(r['log_ratio_ci'][0]):6.3f},{np.exp(r['log_ratio_ci'][1]):6.3f}] "
              f"{r['p_lean_not_better']:6.4f} {'*' if r['holm_better'] else ' '} | {r['lean_at']}")
    json.dump(rows, open(f"{d}/analysis.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10000)
