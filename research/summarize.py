"""Summarise evaluate.py results: worst-case MSE ratio to sampling over all budgets B >= 1.

fair:   at budget B, run the last rung of the fixed ladder const -> roots_k -> any_k -> exact
        whose mean cost is <= B (rungs costing more than exact are dropped, since exact is then
        affordable). It never looks at the truth. This is the primary metric.
oracle: at budget B, use the rung with the smallest MSE among those with mean cost <= B.

The ratio is (MSE of the rung in use) / (sampling MSE = mean_k p_k (1 - p_k) / B). Between rung
costs it increases in B, so its supremum is attained just below the next rung's cost: we compute
it exactly there (no budget grid). Uncertainty: bootstrap over the noisy models.

Usage: python summarize.py results/test/*.jsonl
"""

import json
import re
import sys

import numpy as np


def order_key(name):
    if name == "const":
        return (0, 0)
    if name == "exact":
        return (3, 0)
    if name == "exact_cheap":  # exact refinement with a cheap measure (evaluate.py rule cheap4)
        return (2, 10**9)
    m = re.match(r"(roots|any)(\d+)", name)
    return (1 if m.group(1) == "roots" else 2, int(m.group(2)))


def load(path):
    rows = [json.loads(l) for l in open(path)]
    rows.sort(key=lambda r: r["seed"])
    names = sorted(rows[0]["rungs"], key=order_key)
    truth = np.array([r["truth"] for r in rows])
    est = np.array([[r["rungs"][k][0] for k in names] for r in rows])
    cost = np.array([[r["rungs"][k][1] for k in names] for r in rows])
    return names, truth, est, cost


def worst_ratios(names, truth, est, cost, idx=None):
    if idx is not None:
        truth, est, cost = truth[idx], est[idx], cost[idx]
    var = float((truth * (1 - truth)).mean())
    mse = ((est - truth[:, None]) ** 2).mean(0)
    c = cost.mean(0)
    ex = names.index("exact")
    keep = [i for i in range(len(names)) if i == ex or c[i] < c[ex]]
    if "exact_cheap" in names:  # likewise drop rungs that cost more than the near-exact rung
        ec = names.index("exact_cheap")
        keep = [i for i in keep if i in (ex, ec) or c[i] < c[ec]]
    rungs = sorted(keep, key=lambda i: order_key(names[i]))
    # fair: ladder order; the rung in use at B is the last one with cost <= B
    fair, fair_at = 0.0, None
    for a, b in zip(rungs, rungs[1:] + [None]):
        hi = c[b] if b is not None else np.inf
        if hi <= 1 or mse[a] == 0:
            continue
        r = mse[a] * hi / var if np.isfinite(hi) else np.inf
        if r > fair:
            fair, fair_at = r, (names[a], float(hi))
    # oracle: best affordable MSE so far, in increasing cost order
    by_cost = sorted(keep, key=lambda i: c[i])
    oracle, best = 0.0, np.inf
    for a, b in zip(by_cost, by_cost[1:] + [None]):
        best = min(best, mse[a])
        hi = c[b] if b is not None else np.inf
        if hi <= 1 or best == 0:
            continue
        oracle = max(oracle, best * hi / var)
    return fair, oracle, fair_at, var, c[ex]


def crossover(names, truth, est, cost, idx=None):
    """The smallest budget B* such that, under the fair rule, the ladder has lower MSE than
    sampling at every budget B >= B* (1 if it wins everywhere). Within a rung's budget interval
    the ratio mse * B / var increases, so the ladder loses on the end of the interval from
    B = var / mse on; B* is the end of the last interval where that happens."""
    if idx is not None:
        truth, est, cost = truth[idx], est[idx], cost[idx]
    var = float((truth * (1 - truth)).mean())
    mse = ((est - truth[:, None]) ** 2).mean(0)
    c = cost.mean(0)
    ex = names.index("exact")
    keep = [i for i in range(len(names)) if i == ex or c[i] < c[ex]]
    if "exact_cheap" in names:
        ec = names.index("exact_cheap")
        keep = [i for i in keep if i in (ex, ec) or c[i] < c[ec]]
    rungs = sorted(keep, key=lambda i: order_key(names[i]))
    b_star = 1.0
    for a, b in zip(rungs, rungs[1:] + [None]):
        hi = c[b] if b is not None else np.inf
        if mse[a] > 0 and var / mse[a] < hi:
            b_star = max(b_star, float(hi))
    return b_star


def bootstrap(names, truth, est, cost, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    K = len(truth)
    vals = np.array([worst_ratios(names, truth, est, cost, rng.integers(0, K, K))[0] for _ in range(n)])
    return np.quantile(vals, [0.05, 0.95]), float((vals > 1).mean())


def summary(path, boot=2000):
    names, truth, est, cost = load(path)
    fair, oracle, at, var, exact_cost = worst_ratios(names, truth, est, cost)
    (lo, hi), p_over = bootstrap(names, truth, est, cost, boot)
    return dict(path=path, K=len(truth), mean_acc=float(truth.mean()), exact_cost=float(exact_cost),
                fair=fair, fair_lo=float(lo), fair_hi=float(hi), p_over=p_over, oracle=oracle,
                binding=at)


if __name__ == "__main__":
    print(f"{'file':42s} {'K':>4s} {'mean acc':>9s} {'exact':>7s} {'worst fair':>11s} {'90% CI':>16s} "
          f"{'P(>1)':>6s} {'oracle':>7s}  binding rung")
    for p in sys.argv[1:]:
        s = summary(p)
        print(f"{p.split('/')[-1]:42s} {s['K']:4d} {s['mean_acc']:9.4f} {s['exact_cost']:7.0f} {s['fair']:11.3f} "
              f"[{s['fair_lo']:6.3f}, {s['fair_hi']:6.3f}] {s['p_over']:6.3f} {s['oracle']:7.3f}  {s['binding']}")
