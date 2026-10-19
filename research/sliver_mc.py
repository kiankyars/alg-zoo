"""Monte Carlo checks of the tie-sliver laws (independent of the cone geometry in sliver.py).

1. `law`: the first-order and order-truncated laws estimated by sampling points on each tie
   hyperplane (zeta ~ N(0, I) on H_pq) and measuring the logit gap Delta and its growth rate g
   off H_pq by finite differences.
2. `decompose`: where the exact error lives. Each wrong input x (sampled from N(0, I_n)) lies in
   an order chamber whose target has two (or one) tie walls. x is in the sliver of a wall if the
   model is wrong all along the segment from x to its foot on that wall (the segment stays in the
   chamber). Wrong inputs in exactly one sliver, in two (counted twice by any sum over walls),
   and in none (not sliver errors) are counted separately. The sum over walls of the true sliver
   measures is then (one) + 2 (two), and the truncated law differs from it only through the
   nonlinearity of the logit gap within the sliver.

Usage: python sliver_mc.py <family:d,n> [law|decompose] [log2 samples]
       python sliver_mc.py table [log2 samples]   (decomposition of every model in DECOMPOSE,
                                                   written to results/sliver_decomposition.json)
"""

import itertools
import math
import sys

import numpy as np

from engine import target_rank
from evaluate import load

PHI0 = 1.0 / math.sqrt(2 * math.pi)


def tie_types(r, n, p, q):
    """(rank of the tied pair's upper slot, {side: (target, competitor)}) as in sliver.tie_frames."""
    out = []
    if r >= 1:
        out.append((r - 1, {+1: (q, p), -1: (p, q)}))
    if r + 1 <= n - 1:
        out.append((r, {+1: (p, q), -1: (q, p)}))
    return out


def law(rnn, task, N, rng, eps=1e-7, h=1e-5):
    """Monte Carlo ((first-order law, standard error), (order-truncated law, standard error))."""
    n, r = rnn.n, target_rank(task, rnn.n)
    first = trunc = var_first = var_trunc = 0.0
    for p, q in itertools.combinations(range(n), 2):
        nu = np.zeros(n)
        nu[p], nu[q] = 1 / math.sqrt(2), -1 / math.sqrt(2)
        X = rng.standard_normal((N, n))
        X -= np.outer(X @ nu, nu)  # standard Gaussian on H_pq
        others = [k for k in range(n) if k not in (p, q)]
        y, O = X[:, p], X[:, others]
        rank = (O > y[:, None]).sum(1)
        above = np.where(O > y[:, None], O, np.inf).min(1, initial=np.inf)
        below = np.where(O < y[:, None], O, -np.inf).max(1, initial=-np.inf)
        limit = math.sqrt(2) * np.minimum(above - y, y - below)  # order limit along +-nu
        wf, wt = np.zeros(N), np.zeros(N)  # per-sample contributions of this tie hyperplane
        for obj_rank, targets in tie_types(r, n, p, q):
            sel = rank == obj_rank
            Xs, lim = X[sel], limit[sel]
            for s in (+1, -1):
                t, c = targets[s]
                L = rnn.forward(Xs)
                La = rnn.forward(Xs + eps * s * nu)
                Lb = rnn.forward(Xs + (eps + h) * s * nu)
                delta = L[:, t] - L[:, c]
                g = ((Lb[:, t] - Lb[:, c]) - (La[:, t] - La[:, c])) / h
                ok = (delta < 0) & (L.argmax(1) == c) & (g > 0)
                w = np.where(ok, -delta / np.where(ok, g, 1.0), 0.0)
                wf[sel] += PHI0 * w
                wt[sel] += PHI0 * np.minimum(w, lim)
        first, var_first = first + wf.mean(), var_first + wf.var() / N
        trunc, var_trunc = trunc + wt.mean(), var_trunc + wt.var() / N
    return (first, math.sqrt(var_first)), (trunc, math.sqrt(var_trunc))


def decompose(rnn, task, N, rng, steps=64):
    """Fractions (of all inputs) of wrong inputs in exactly one / two / no tie slivers."""
    n = rnn.n
    r = target_rank(task, n)
    X = rng.standard_normal((N, n))
    order = np.argsort(-X, 1)
    target = order[:, r]
    wrong = rnn.forward(X).argmax(1) != target
    X, order, target = X[wrong], order[wrong], target[wrong]
    idx = np.arange(len(X))
    inside = []
    for a, b in ((r - 1, r), (r, r + 1)):  # the target's walls (ranks a, b tied)
        if a < 0 or b > n - 1:
            continue
        u, v = order[:, a], order[:, b]
        half = (X[idx, u] - X[idx, v]) / 2  # foot: both moved to their mean
        ok = np.ones(len(X), bool)
        for k in range(1, steps + 1):  # every point of the segment (foot excluded) is wrong
            lam = k / steps
            Y = X.copy()
            Y[idx, u] -= (1 - lam) * half
            Y[idx, v] += (1 - lam) * half
            ok &= rnn.forward(Y).argmax(1) != target
        inside.append(ok)
    count = np.sum(inside, axis=0)
    return [(count == k).sum() / N for k in (1, 2, 0)], wrong.mean()


DECOMPOSE = ["M:6,3", "M:8,3", "M:16,3", "M:32,3", "M:6,4", "M:8,4", "M:16,4", "M:32,4",
             "M:6,5", "M:8,5", "M:16,5", "M:32,5", "AM:8,5", "AM:16,5", "AM:32,5"]


if __name__ == "__main__":
    if sys.argv[1] == "table":
        import json

        N = 1 << int(sys.argv[2] if len(sys.argv) > 2 else 22)
        rows = []
        for spec in DECOMPOSE:
            rnn, (task, _) = load(spec)
            (one, two, none), err = decompose(rnn, task, N, np.random.default_rng(0))
            fam, rest = spec.split(":")
            d, n = map(int, rest.split(","))
            rows.append(dict(family=fam, d=d, n=n, samples=N, error=err, one=one, two=two, none=none))
            print(f"{spec}: error {err:.4e}: one {one / err:.1%}, two {two / err:.1%}, none {none / err:.1%}",
                  flush=True)
        json.dump(rows, open("results/sliver_decomposition.json", "w"), indent=1)
        sys.exit()
    spec = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "law"
    N = 1 << int(sys.argv[3] if len(sys.argv) > 3 else 20)
    rnn, (task, _) = load(spec)
    rng = np.random.default_rng(0)
    if mode == "law":
        (first, se1), (trunc, se2) = law(rnn, task, N, rng)
        print(f"{spec}: Monte Carlo first-order law {first:.4e} ± {se1:.1e}, "
              f"order-truncated law {trunc:.4e} ± {se2:.1e}")
    else:
        (one, two, none), err = decompose(rnn, task, N, rng)
        print(f"{spec}: error {err:.4e} = in one sliver {one:.4e} + in two slivers {two:.4e} "
              f"+ in no sliver {none:.4e}   (sum over walls of true slivers {one + 2 * two:.4e})")
