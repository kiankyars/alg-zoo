"""Pre-registered analysis of the held-out test of the cheap-measure ladder for 4-input models
(PREREG_cheap4.md): the cheap4 and lguard ladders on the same noisy models (one tag directory
written by evaluate.py).

For each cell (spec, sigma) with both <cell>_cheap4.jsonl and <cell>_lguard.jsonl:
  * checks that both files hold the same seeds and the same truth, and that the exact rung is
    the truth;
  * primary: the cheap4 ladder's worst fair ratio over budgets B >= b0 (summarize.worst_from),
    with b0 fixed in advance per model (B0 below), p = P*(ratio >= 1) from a paired percentile
    bootstrap (10,000 replicates, seed 0), Holm at 0.05 over the cells;
  * secondary (descriptive, 90% intervals): each ladder's worst fair ratio over all budgets,
    their paired ratio (Holm over the cells for "cheap4 smaller"), and each ladder's crossover
    budget B* (summarize.crossover: from B* on it beats sampling at every budget).

Usage: python analyze_test4.py results/test8000 [n_boot]   (also writes <dir>/analysis.json)
"""

import glob
import json
import os
import sys

import numpy as np

from analyze_test import holm
from summarize import crossover, load, worst_from, worst_ratios

B0 = {"M_8_4": 6.3e4, "M_16_4": 1.35e5}  # forward passes; 1.1 x the dev cost of exact_cheap


def main(d, n_boot=10000):
    cells = sorted({os.path.basename(f).rsplit("_", 1)[0] for f in glob.glob(f"{d}/*.jsonl")},
                   key=lambda c: (int(c.split("_")[1]), float(c.split("_")[3])))
    rows = []
    for c in cells:
        fa, fb = f"{d}/{c}_cheap4.jsonl", f"{d}/{c}_lguard.jsonl"
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
        b0 = B0[c.rsplit("_", 1)[0]]
        K = len(A[1])
        rng = np.random.default_rng(0)
        bs = np.empty((n_boot, 5))
        for i in range(n_boot):
            idx = rng.integers(0, K, K)
            bs[i] = (worst_from(*A, b0, idx), worst_ratios(*A, idx)[0], worst_ratios(*B, idx)[0],
                     crossover(*A, idx), crossover(*B, idx))
        q = lambda j: np.quantile(bs[:, j], [0.05, 0.95]).tolist()
        wa, wb = worst_ratios(*A), worst_ratios(*B)
        lr = np.log(bs[:, 1] / bs[:, 2])
        rows.append(dict(cell=c, K=K, mean_acc=float(A[1].mean()), b0=b0,
                         exact_cost=float(A[3][:, A[0].index("exact")].mean()),
                         from_b0=worst_from(*A, b0), from_b0_ci=q(0), p_from_b0=float((bs[:, 0] >= 1).mean()),
                         cheap4=wa[0], cheap4_at=wa[2], cheap4_ci=q(1),
                         lguard=wb[0], lguard_at=wb[2], lguard_ci=q(2),
                         log_ratio=float(np.log(wa[0] / wb[0])), log_ratio_ci=np.quantile(lr, [0.05, 0.95]).tolist(),
                         p_cheap4_not_better=float((lr >= 0).mean()),
                         crossover_cheap4=crossover(*A), crossover_cheap4_ci=q(3),
                         crossover_lguard=crossover(*B), crossover_lguard_ci=q(4)))
    if not rows:
        return
    for key, fam in (("p_from_b0", "holm_from_b0"), ("p_cheap4_not_better", "holm_better")):
        for r, rej in zip(rows, holm([r[key] for r in rows])):
            r[fam] = bool(rej)
    print(f"{len(rows)} cells, {n_boot} paired bootstrap replicates, Holm at 0.05 within each family")
    for r in rows:
        print(f"{r['cell']:12s} K={r['K']} acc {r['mean_acc']:.4f} exact {r['exact_cost']:.3g} | "
              f"worst over B >= {r['b0']:.2g}: {r['from_b0']:.4f} [{r['from_b0_ci'][0]:.4f},{r['from_b0_ci'][1]:.4f}] "
              f"p {r['p_from_b0']:.4f} {'*' if r['holm_from_b0'] else ' '} | "
              f"all budgets: cheap4 {r['cheap4']:.1f} [{r['cheap4_ci'][0]:.1f},{r['cheap4_ci'][1]:.1f}] "
              f"lguard {r['lguard']:.0f} [{r['lguard_ci'][0]:.0f},{r['lguard_ci'][1]:.0f}] "
              f"ratio {np.exp(r['log_ratio']):.3f} {'*' if r['holm_better'] else ' '} | "
              f"B* cheap4 {r['crossover_cheap4']:.3g} [{r['crossover_cheap4_ci'][0]:.3g},{r['crossover_cheap4_ci'][1]:.3g}] "
              f"lguard {r['crossover_lguard']:.3g}")
    json.dump(rows, open(f"{d}/analysis.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10000)
