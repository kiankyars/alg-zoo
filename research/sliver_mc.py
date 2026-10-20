"""Monte Carlo checks of the tie-sliver laws (independent of the cone geometry in sliver.py).

1. `law`: the first-order and order-truncated laws estimated by sampling points on each tie
   hyperplane (zeta ~ N(0, I) on H_pq) and measuring the logit gap Delta and its growth rate g
   off H_pq by finite differences.
2. `decompose`: where the exact error lives. Each wrong input x (sampled from N(0, I_n)) lies in
   an order chamber whose target has two (or one) tie walls. x is in the sliver of a wall if the
   model is wrong all along the segment from x to its foot on that wall (the segment stays in the
   chamber). Wrong inputs in exactly one sliver, in two (counted twice by any sum over walls),
   and in none (not sliver errors) are counted separately. The sum over walls of the true slivers
   is then (one) + 2 (two), and the truncated law differs from it only through the nonlinearity
   of the logit gap within the sliver.
3. `widths`: the sliver laws with the true (contiguous) width along each normal instead of the
   first-order width, found by a geometric grid and bisection, with the chamber cap and with the
   bisector cap (each wrong input belongs to the nearer of its target's two walls), and the
   share of wrong inputs that are not in the sliver of their nearer target wall.

Usage: python sliver_mc.py <family:d,n> [law|decompose|widths] [log2 samples]
       python sliver_mc.py table [log2 samples]   (decomposition of every model in DECOMPOSE,
                                                   written to results/sliver_decomposition.json)
       python sliver_mc.py widths-table [log2 samples per tie] [specs]   (Monte Carlo exact sliver
                                                   law of WIDTHS, written to results/sliver_widths.json)

The grid checks (decompose, unattached, the grid stage of true_widths) can only miss short correct
stretches between grid points, so they can only understate the share of wrong inputs that are not
in a sliver; true_widths pins each transition down by bisection.
"""

import itertools
import math
import sys

import numpy as np

from engine import target_rank
from evaluate import load

PHI0 = 1.0 / math.sqrt(2 * math.pi)


def tie_types(r, n, p, q):
    """(rank of the tied pair's upper slot, {side: (target, competitor)}, target-wall limit)
    as in sliver.tie_frames. The target-wall limit says which order limit is the target's own
    crossing into its other target wall: "down" when the target is the falling tied value."""
    out = []
    if r >= 1:
        out.append((r - 1, {+1: (q, p), -1: (p, q)}, "down"))
    if r + 1 <= n - 1:
        out.append((r, {+1: (p, q), -1: (q, p)}, "up"))
    return out


def law(rnn, task, N, rng, eps=1e-7, h=1e-5, bisector=False):
    """Monte Carlo ((first-order law, standard error), (order-truncated law, standard error)), and
    with bisector=True also (first-order law with the bisector cap, standard error)."""
    n, r = rnn.n, target_rank(task, rnn.n)
    first = trunc = var_first = var_trunc = bis = var_bis = 0.0
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
        up, down = math.sqrt(2) * (above - y), math.sqrt(2) * (y - below)
        wf, wt, wb = np.zeros(N), np.zeros(N), np.zeros(N)  # per-sample contributions of this tie
        for obj_rank, targets, tlim in tie_types(r, n, p, q):
            sel = rank == obj_rank
            Xs, lim = X[sel], limit[sel]
            T, U = (down[sel], up[sel]) if tlim == "down" else (up[sel], down[sel])
            blim = np.minimum(T / 3, U)
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
                wb[sel] += PHI0 * np.minimum(w, blim)
        first, var_first = first + wf.mean(), var_first + wf.var() / N
        trunc, var_trunc = trunc + wt.mean(), var_trunc + wt.var() / N
        bis, var_bis = bis + wb.mean(), var_bis + wb.var() / N
    out = (first, math.sqrt(var_first)), (trunc, math.sqrt(var_trunc))
    return out + ((bis, math.sqrt(var_bis)),) if bisector else out


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


def true_widths(rnn, X0, direction, target, cap, ratio=1.02, tau0=1e-9, bisect=40):
    """Length of the wrong stretch of each line X0 + tau * direction that starts at tau = 0+,
    capped at `cap`: the first tau at which the prediction is `target` (a geometric grid finds
    the first correct grid point, then bisection pins down the transition), or `cap`."""
    M = len(X0)
    grid = np.concatenate([[0.0], tau0 * ratio ** np.arange(int(math.log(4 / tau0) / math.log(ratio)) + 2)])
    lo = np.zeros(M)  # last known wrong point
    hi = np.full(M, np.inf)  # first known correct point
    alive = rnn.forward(X0 + tau0 * 1e-3 * direction).argmax(1) != target
    hi[~alive] = 0.0
    for a, b in zip(grid[:-1], grid[1:]):
        act = alive & (a < cap)
        if not act.any():
            break
        ok = rnn.forward(X0[act] + b * direction).argmax(1) == target[act]
        idx = np.where(act)[0]
        hi[idx[ok]] = b
        lo[idx[~ok]] = b
        alive[idx[ok]] = False
    found = np.isfinite(hi) & (hi > 0)
    for _ in range(bisect):
        mid = (lo + hi) / 2
        idx = np.where(found)[0]
        if len(idx) == 0:
            break
        ok = rnn.forward(X0[idx] + mid[idx, None] * direction).argmax(1) == target[idx]
        hi[idx[ok]] = mid[idx[ok]]
        lo[idx[~ok]] = mid[idx[~ok]]
    w = np.where(np.isfinite(hi), (lo + hi) / 2, np.inf)
    w[hi == 0] = 0.0
    return np.minimum(w, cap)


def widths(rnn, task, N, rng):
    """Monte Carlo of the sliver laws with true widths: dict with the chamber-capped and the
    bisector-capped sums (each with its standard error)."""
    n, r = rnn.n, target_rank(task, rnn.n)
    out = {}
    for name in ("chamber", "bisector"):
        out[name] = [0.0, 0.0]
    for p, q in itertools.combinations(range(n), 2):
        nu = np.zeros(n)
        nu[p], nu[q] = 1 / math.sqrt(2), -1 / math.sqrt(2)
        X = rng.standard_normal((N, n))
        X -= np.outer(X @ nu, nu)
        others = [k for k in range(n) if k not in (p, q)]
        y, O = X[:, p], X[:, others]
        rank = (O > y[:, None]).sum(1)
        up = math.sqrt(2) * (np.where(O > y[:, None], O, np.inf).min(1, initial=np.inf) - y)
        down = math.sqrt(2) * (y - np.where(O < y[:, None], O, -np.inf).max(1, initial=-np.inf))
        acc = {k: np.zeros(N) for k in out}
        for obj_rank, targets, tlim in tie_types(r, n, p, q):
            sel = rank == obj_rank
            Xs = X[sel]
            T, U = (down[sel], up[sel]) if tlim == "down" else (up[sel], down[sel])
            caps = {"chamber": np.minimum(T, U), "bisector": np.minimum(T / 3, U)}
            for s in (+1, -1):
                t, _ = targets[s]
                tgt = np.full(len(Xs), t)
                w = true_widths(rnn, Xs, s * nu, tgt, caps["chamber"])
                for k, cap in caps.items():
                    acc[k][sel] += PHI0 * np.minimum(w, cap)
        for k in out:
            out[k][0] += acc[k].mean()
            out[k][1] += acc[k].var() / N
    return {k: (v[0], math.sqrt(v[1])) for k, v in out.items()}


def unattached(rnn, task, N, rng, steps=64):
    """Fraction (of all inputs) of wrong inputs that are not in the sliver of the nearer of their
    target's two walls (the remainder of the bisector-capped decomposition), and the error."""
    n = rnn.n
    r = target_rank(task, n)
    X = rng.standard_normal((N, n))
    order = np.argsort(-X, 1)
    target = order[:, r]
    wrong = rnn.forward(X).argmax(1) != target
    X, order, target = X[wrong], order[wrong], target[wrong]
    idx = np.arange(len(X))
    dist, walls = [], []
    for a, b in ((r - 1, r), (r, r + 1)):
        if a < 0 or b > n - 1:
            continue
        u, v = order[:, a], order[:, b]
        dist.append((X[idx, u] - X[idx, v]) / math.sqrt(2))
        walls.append((u, v))
    near = np.argmin(np.stack(dist), 0) if len(dist) > 1 else np.zeros(len(X), int)
    ok = np.ones(len(X), bool)
    for wi, (u, v) in enumerate(walls):
        sel = near == wi
        uu, vv, ii = u[sel], v[sel], idx[sel]
        half = (X[ii, uu] - X[ii, vv]) / 2
        good = np.ones(sel.sum(), bool)
        for k in range(1, steps + 1):
            lam = k / steps
            Y = X[ii].copy()
            Y[np.arange(len(ii)), uu] -= (1 - lam) * half
            Y[np.arange(len(ii)), vv] += (1 - lam) * half
            good &= rnn.forward(Y).argmax(1) != target[ii]
        ok[ii] = good
    return (~ok).sum() / N, wrong.mean()


DECOMPOSE = ["M:3,3", "M:4,3", "M:5,3", "M:6,3", "M:8,3", "M:16,3", "M:32,3", "M:6,4", "M:8,4", "M:16,4",
             "M:32,4", "M:6,5", "M:8,5", "M:16,5", "M:32,5", "AM:8,5", "AM:16,5", "AM:32,5"]


WIDTHS = ["M:32,5", "AM:32,5", "M:6,5", "M:8,5", "AM:8,5"]  # sliver_table.NO_EXACT: too costly to compute exactly


if __name__ == "__main__":
    if sys.argv[1] == "widths-table":  # Monte Carlo exact sliver law (bisector cap, true widths)
        import json

        N = 1 << int(sys.argv[2] if len(sys.argv) > 2 else 17)
        rows = []
        for spec in sys.argv[3:] or WIDTHS:
            rnn, (task, _) = load(spec)
            res = widths(rnn, task, N, np.random.default_rng(0))
            fam, rest = spec.split(":")
            d, n = map(int, rest.split(","))
            rows.append(dict(family=fam, d=d, n=n, samples_per_tie=N, exact_sliver=res["bisector"][0],
                             exact_sliver_se=res["bisector"][1], chamber_true=res["chamber"][0],
                             chamber_true_se=res["chamber"][1]))
            print(f"{spec}: Monte Carlo exact sliver law {res['bisector'][0]:.4e} ± {res['bisector'][1]:.1e}",
                  flush=True)
        json.dump(rows, open("results/sliver_widths.json", "w"), indent=1)
        sys.exit()
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
    elif mode == "widths":
        res = widths(rnn, task, N, rng)
        print(f"{spec}: Monte Carlo true-width laws: " +
              ", ".join(f"{k} {v:.4e} ± {se:.1e}" for k, (v, se) in res.items()))
    elif mode == "unattached":
        un, err = unattached(rnn, task, N, rng)
        print(f"{spec}: error {err:.4e}, not in the nearer wall's sliver {un:.3e} ({un / err:.2%})")
    else:
        (one, two, none), err = decompose(rnn, task, N, rng)
        print(f"{spec}: error {err:.4e} = in one sliver {one:.4e} + in two slivers {two:.4e} "
              f"+ in no sliver {none:.4e}   (sum over walls of true slivers {one + 2 * two:.4e})")
