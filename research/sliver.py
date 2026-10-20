"""First-order "tie sliver" estimate of the error of an order-statistic RNN classifier.

Mechanism. A well-trained 2nd-argmax (or argmedian) RNN errs almost exclusively in thin wedges
("slivers") along the tie hyperplanes H_pq = {x_p = x_q} at which the target switches between
positions p and q (ties between the target rank and an adjacent rank). On H_pq the logits of p
and q should be equal; the model's actual logit gap there, Delta(x) = l_t(x) - l_c(x) between
the side's target t and its competitor c, is what displaces its decision boundary off the tie
hyperplane. Moving a distance tau off H_pq into side s, the gap changes at rate g, so the model
is wrong for 0 < tau < -Delta(x) / g whenever Delta(x) < 0 < g. To first order in the sliver
width the error measure is therefore

    sum over ties, sides, linear pieces K of the model on H_pq:
        phi(0) / g_K * E[-Delta_K . zeta ; zeta in K, Delta_K . zeta < 0]

with zeta a standard Gaussian on H_pq (dimension n - 1). By Gaussian integration by parts
E[zeta 1_K] = phi(0) sum_facets F (unit inward normal of F) * gamma_{n-2}(F), so the estimate
only needs (n - 2)-dimensional cone measures, which are closed form for n <= 5. Only the
(n - 1)-dimensional tie hyperplanes need to be refined into linear pieces, not R^n.

Pieces where the first-order picture does not apply (another class is the argmax at the tie,
or g <= 0 so the error is not thin) are reported separately as `flagged` H-measure.
"""

import itertools
import math

import numpy as np

from engine import TOL, Cone, Flops, _maximal, _pull, split, target_rank, triangulate
from orthant import cone_measure
from slice import slice_flops, slice_linear_moment, slice_sliver_measure

PHI0 = 1.0 / math.sqrt(2 * math.pi)


def tie_frames(n, task):
    """Yields (p, q, E, nu, roots, limits, targets): E (n, n-1) orthonormal basis of H_pq whose
    first column is (e_p + e_q)/sqrt 2, nu = (e_p - e_q)/sqrt 2, roots = simplicial cones (rays
    in zeta coordinates) covering the part of H_pq where the tie is between the target rank and
    an adjacent rank, limits[i] = dict(target=..., other=...): the order limits of roots[i] (see
    below; None if absent), and targets = {side: (target, competitor)} for side in (+1, -1).

    Order limits. Moving a distance tau off H_pq along +-nu, one tied value rises and the other
    falls by tau / sqrt 2, so the line leaves the order chamber of the tie face (the ordering
    changes) once the riser passes the object just above the tie, at
    tau = sqrt 2 x_above - zeta_0, or the faller passes the one just below, at
    tau = zeta_0 - sqrt 2 x_below. Within a root the ordering is fixed, so both are linear
    functionals of zeta (nonnegative on the root). The "target" limit is the target's own crossing,
    where the line reaches the target's other wall (the other tie of the target with an adjacent
    rank); the "other" limit is the competitor's crossing, where the line leaves the chamber
    through a wall at which the target does not change."""
    r = target_rank(task, n)
    m = n - 1
    for p, q in itertools.combinations(range(n), 2):
        others = [k for k in range(n) if k not in (p, q)]
        E = np.zeros((n, m))
        E[p, 0] = E[q, 0] = 1 / math.sqrt(2)
        for j, k in enumerate(others):
            E[k, j + 1] = 1.0
        nu = np.zeros(n)
        nu[p], nu[q] = 1 / math.sqrt(2), -1 / math.sqrt(2)
        # tie between ranks (r-1, r): tied object at rank r-1; side + means x_p > x_q so p has
        # rank r-1 and the target is q. Tie between ranks (r, r+1): tied object at rank r.
        types = []
        if r >= 1:
            types.append((r - 1, {+1: (q, p), -1: (p, q)}))
        if r + 1 <= n - 1:
            types.append((r, {+1: (p, q), -1: (q, p)}))
        for obj_rank, targets in types:
            roots, limits = [], []
            for perm in itertools.permutations(range(m)):
                # perm[i] = object at sorted rank i; object 0 is the tie, object j>0 = others[j-1]
                if perm[obj_rank] != 0:
                    continue
                up = down = None
                if obj_rank >= 1:  # sqrt 2 zeta_above - zeta_0
                    up = math.sqrt(2) * np.eye(m)[perm[obj_rank - 1]] - np.eye(m)[0]
                if obj_rank + 1 <= m - 1:  # zeta_0 - sqrt 2 zeta_below
                    down = np.eye(m)[0] - math.sqrt(2) * np.eye(m)[perm[obj_rank + 1]]
                # the target is the faller for ties (r-1, r) and the riser for ties (r, r+1): its
                # own crossing leads to its other target wall, the competitor's out of the chamber
                lim = dict(target=down, other=up) if obj_rank == r - 1 else dict(target=up, other=down)
                for k in range(m + 1):
                    Y = np.zeros((m, m))  # rows: rays in sorted object coordinates
                    for j in range(1, k + 1):
                        Y[j - 1, :j] = 1.0
                    for j in range(1, m - k + 1):
                        Y[k + j - 1, m - j :] = -1.0
                    Z = np.zeros((m, m))
                    Z[:, list(perm)] = Y  # object values
                    Z[:, 0] *= math.sqrt(2)  # zeta_0 = sqrt(2) * tied value
                    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
                    roots.append(Z)
                    limits.append(lim)
            yield p, q, E, nu, roots, limits, targets


def tie_pieces(rnn, cone, E, side_dir, fl):
    """Linear pieces of the RNN on a cone in H (zeta coordinates), taking for each neuron whose
    pre-activation vanishes identically on the piece the sign it has just off H in direction
    side_dir (the one-sided limit). Yields (cone, L, masks) with L the logits map in x
    coordinates and masks (n, d) the activation pattern of the piece."""
    d, n = rnn.d, rnn.n
    m = E.shape[1]
    # stack items: (cone, t, H, (P, PE, sc) or None, S); as in engine.linear_pieces, children
    # inherit the step-t pre-activation map and their parent's ray pre-activations, so only
    # the new rays of a split are evaluated
    stack = [(cone, 0, np.zeros((d, n)), None, None, ())]
    while stack:
        cone, t, H, maps, S, masks = stack.pop()
        split_done = False
        while t < n:
            if maps is None:
                P = rnn.Wh @ H
                P[:, t] += rnn.Wi
                PE = P @ E  # the pre-activation map in zeta coordinates
                sc = np.linalg.norm(P, axis=1) + 1e-300
                maps = (P, PE, sc)
                S = cone.V @ PE.T
                fl.rnn += 2 * d * d * t + 2 * d * (t + 1) * m + d + 2 * len(S) * m * d
            P, PE, sc = maps
            pos = S > TOL * sc
            neg = S < -TOL * sc
            fl.rnn += 4 * len(S) * d + d
            conflict = pos.any(0) & neg.any(0)
            if conflict.any():
                i = np.where(conflict)[0][0]
                plus, minus, (Pi, Ni, Zi, W) = split(cone, PE[i], fl, parts=True)
                SW = W @ PE.T
                fl.rnn += 2 * len(W) * m * d
                stack.append((plus, t, H, maps, np.concatenate([S[Pi], S[Zi], SW]), masks))
                stack.append((minus, t, H, maps, np.concatenate([S[Ni], S[Zi], SW]), masks))
                split_done = True
                break
            zero = ~pos.any(0) & ~neg.any(0)
            mask = pos.any(0) | (zero & (P @ side_dir > TOL * sc))
            H = mask[:, None] * P
            masks = masks + (mask,)
            fl.rnn += 2 * d * n + d * (t + 1)
            t += 1
            maps = S = None
        if not split_done:
            fl.rnn += 2 * len(rnn.Wo) * d * n
            yield cone, rnn.Wo @ H, np.array(masks)


def facet_measures(cone, fl):
    """(unit inward normals (k, m), (m-1)-dim Gaussian measures (k,)) of the facets of a cone in
    R^m, the latter computed in each facet's own hyperplane."""
    m = cone.V.shape[1]
    facets = [f for f in range(cone.I.shape[1]) if cone.I[:, f].sum() >= m - 1]
    normals = cone.F[facets]
    if m <= 2:  # facets are the origin (of 0-dim measure 1) or rays (of 1-dim measure 1/2)
        return normals, np.full(len(facets), 1.0 if m == 1 else 0.5)
    all_sets = [frozenset(np.where(cone.I[:, f])[0].tolist()) for f in facets]
    blocks, owner = [], []
    for k, (nrm, S) in enumerate(zip(normals, all_sets)):
        if len(S) == m - 1:
            simps = [tuple(sorted(S))]
        else:
            ridges = _maximal(S & T for T in all_sets if T is not S and len(S & T) >= m - 2)
            simps = _pull(sorted(S), ridges, m - 1)
        # coordinates in the facet hyperplane: a Householder reflection maps nrm to +-e_j,
        # and the remaining m - 1 coordinates are an orthonormal basis of the hyperplane
        j = int(np.abs(nrm).argmax())
        w = nrm.copy()
        w[j] += math.copysign(1.0, nrm[j])
        w /= np.linalg.norm(w)
        R = cone.V[np.array(simps)]
        R = np.delete(R - 2 * (R @ w)[..., None] * w, j, axis=-1)  # (s, m-1, m-1)
        blocks.append(R)
        owner += [k] * len(simps)
        fl.measure += len(simps) * (60 + 4 * m**3)
    meas = np.zeros(len(facets))
    np.add.at(meas, owner, cone_measure(np.concatenate(blocks)))
    return normals, meas


def first_moment(cone, fl):
    """E[zeta 1_K(zeta)] for zeta ~ N(0, I_m), via Gaussian integration by parts."""
    N, g = facet_measures(cone, fl)
    return PHI0 * (N * g[:, None]).sum(0)


def linear_moment(cone, w, fl):
    """E[(w . zeta) 1_K(zeta)] for zeta ~ N(0, I_m) and a linear functional w (nonnegative on K).

    This equals w . E[zeta 1_K] (first_moment), but that is ill-conditioned on flat pieces: when
    the logit gap Delta is large and its growth rate g tiny, a piece next to the zero set of
    Delta can be a wedge of thickness ~1e-8, whose first moment comes from nearly cancelling
    facet measures, and the width -Delta / g multiplies that rounding error by up to 1e8 (it
    produced spurious contributions of ~5e-6 in M_{32,5}). So K is triangulated and integrated
    by slice cubature (slice.slice_linear_moment), which stays accurate on flat cones."""
    m = cone.V.shape[1]
    if m == 1:
        return float(w @ first_moment(cone, fl))
    vals, pieces = slice_linear_moment(triangulate(cone), w, return_pieces=True)
    fl.split += 5 * len(cone.V) * m
    fl.measure += slice_flops(m, 3) * pieces
    return float(vals.sum())


def min_moment(K, widths, fl):
    """E[min_i (w_i . zeta) 1_K(zeta)] for linear functionals w_i that are nonnegative on K: K is
    split into the cones on which each w_i is the smallest (ties go to the first), each
    contributing E[(w_i . zeta) 1] (linear_moment)."""
    W = np.stack(widths)
    vals = K.V @ W.T  # (rays, functionals); a linear inequality holds on K iff it holds at the rays
    fl.split += 2 * W.size * len(K.V)
    if (vals[:, :1] <= vals[:, 1:]).all():  # the usual case: the first is the smallest on all of K
        return linear_moment(K, widths[0], fl)
    out = 0.0
    for i, w in enumerate(widths):
        Ki = K
        for j, v in enumerate(widths):
            if j == i or Ki is None:
                continue
            if np.abs(v - w).max() <= 1e-12 * np.abs(w).max():
                if j < i:
                    Ki = None
                continue
            Ki, _ = split(Ki, v - w, fl)  # where w_i <= w_j
        if Ki is not None:
            out += linear_moment(Ki, w, fl)
    return out


def _caps(lim, cap):
    """The width caps of a root as linear functionals: none, the chamber boundary (both order
    limits) or the nearer-wall boundary (the bisector between the target's two walls, at a third
    of the target's own limit, and the chamber exit through the competitor's crossing)."""
    T, U = lim["target"], lim["other"]
    if cap == "chamber":
        return [c for c in (T, U) if c is not None]
    if cap == "bisector":
        return [c for c in (None if T is None else T / 3, U) if c is not None]
    return []


def _pattern_maps(rnn, masks):
    """Pre-activation maps P_t (d, n) of every step and the logits map (n_out, n) of the linear
    region with activation pattern masks (n, d)."""
    d, n = rnn.d, rnn.n
    H = np.zeros((d, n))
    maps = []
    for t in range(n):
        P = rnn.Wh @ H
        P[:, t] += rnn.Wi
        maps.append(P)
        H = masks[t][:, None] * P
    return maps, rnn.Wo @ H


def _sign(Q, f):
    """+1 if the linear functional f is >= 0 on the cone Q, -1 if it is <= 0 (and not ~0), 0 if
    it changes sign (with the same tolerance as engine.split, so split(Q, f) then cuts Q)."""
    nrm = np.linalg.norm(f)
    if nrm == 0:
        return 1
    v = Q.V @ (f / nrm)
    pos, neg = (v > TOL).any(), (v < -TOL).any()
    return 0 if pos and neg else (-1 if neg else 1)


class _Slivers:
    """Collects sliver cones (triangulated: simplicial cones and their width functionals) and
    measures them in batches (slice.slice_sliver_measure: exact Gaussian measure of
    {zeta + tau nu : zeta in K, 0 <= tau <= w . zeta}, including the decay of the density
    across the sliver)."""

    def __init__(self, fl, batch=4096):
        self.fl, self.batch, self.total = fl, batch, 0.0
        self.T, self.W, self.size = [], [], 0

    def add(self, Q, w):
        m = Q.V.shape[1]
        if len(Q.V) < m:  # lower-dimensional (degenerate) cone: measure zero
            return 0.0
        T = triangulate(Q)
        if len(T) == 0:  # no simplices: degenerate
            return 0.0
        self.fl.split += 5 * len(Q.V) * m
        self.T.append(T)
        self.W.append(np.broadcast_to(w, (len(T), m)))
        self.size += len(T)
        if self.size >= self.batch:
            self.flush()
        return 0.0

    def flush(self):
        if self.T:
            T = np.concatenate(self.T)
            vals, pieces = slice_sliver_measure(T, np.concatenate(self.W), return_pieces=True)
            self.fl.measure += slice_flops(T.shape[1], 3) * pieces
            self.total += float(vals.sum())
            self.T, self.W, self.size = [], [], 0
        return self.total


def _signs(Q, F):
    """Row-wise _sign for a matrix of functionals F (k, m): +1 (>= 0 on Q), -1 (<= 0, not ~0) or
    0 (changes sign on Q)."""
    nrm = np.linalg.norm(F, axis=1)
    v = (Q.V @ F.T) / np.where(nrm > 0, nrm, 1.0)
    pos, neg = (v > TOL).any(0), (v < -TOL).any(0)
    return np.where(pos & neg, 0, np.where(neg, -1, 1))


REL = 1e-9  # differences of functionals below REL times their size are rounding noise


def _denoise(F, scale):
    """F (..., m) with every functional whose norm is at most REL * scale set to zero. A difference
    of two functionals that are equal in exact arithmetic (a gap that vanishes where a neuron
    flips, the gap's root and that flip, two gaps proportional to one pre-activation) is rounding
    noise; normalised by its own norm it would get an arbitrary sign of order one."""
    F = np.array(F, dtype=np.float64)
    F[np.linalg.norm(F, axis=-1) <= REL * np.asarray(scale)] = 0.0
    return F


def _walk(rnn, Q, masks, E, sdir, t, caps, fl, acc, tau0=None, depth=0, cache=None, max_depth=200):
    """Adds to the accumulator acc (_Slivers) the true sliver over Q: along each normal line
    zeta + tau sdir (zeta in Q, a piece of the tie hyperplane with activation pattern masks just
    off it), the stretch 0 < tau < w(zeta) on which the prediction is not the target t, where
    w(zeta) is the first tau at which t is the predicted class, or the first cap. The prediction
    is the argmax of the logits with ties going to the lowest index, as in engine.

    The line is followed exactly through the linear regions it crosses. In the current region
    every logit gap f_j = l_t - l_j and every pre-activation is linear in (zeta, tau), so the
    first neuron flip, the first point where t wins and the caps are linear functionals of zeta.
    Q is split until the order of these events is the same on all of it (a linear inequality
    holds on a cone iff it holds at its rays). If a cap or the target winning comes first, the
    sliver ends there; if a neuron flips first, the walk continues in the neighbouring region
    (that neuron's sign toggled; continuity keeps every other sign at the crossing), from the
    crossing time tau0 (also linear in zeta). A flip at the same time as the target's win is
    taken first: there the gaps can vanish together (they are then proportional to that neuron's
    pre-activation) and the next region decides. A gap that vanishes at entry on all of Q takes
    the sign it has just after entry (its slope along the line); a gap that vanishes on the whole
    region (identical logits) follows the tie rule. A win at single points of the line (the
    target's interval [lo, hi] reduced to a point) does not end the sliver. Functionals that are
    equal in exact arithmetic are compared as equal (_denoise)."""
    m = Q.V.shape[1]
    if tau0 is None:
        tau0 = np.zeros(m)
    if cache is None:
        cache = {}
    if depth > max_depth:
        raise RuntimeError("sliver walk did not terminate")
    key = masks.tobytes()
    if key not in cache:
        maps, L = _pattern_maps(rnn, masks)
        fl.rnn += 2 * rnn.d * rnn.d * rnn.n + 2 * len(L) * rnn.d * rnn.n
        D = L[t] - L  # (n_out, n): f_j(x) = D_j . x
        tied = np.abs(D).max(1) <= 1e-12 * (np.abs(L).max() + 1e-300)  # l_j = l_t on the region
        Pall = np.concatenate(maps)  # (n d, n): every pre-activation map
        on = masks.reshape(-1)
        rate = Pall @ sdir
        moving = np.where((on & (rate < 0)) | (~on & (rate > 0)))[0]
        flips = -(Pall[moving] @ E) / rate[moving, None]  # pre-activation zero at tau = flip . zeta
        cache[key] = (D @ E, D @ sdir, tied, moving, flips)
    a, b, tied, moving, flips = cache[key]
    again = lambda K, **kw: _walk(rnn, K, kw.get("masks", masks), E, sdir, t, caps, fl, acc,
                                  kw.get("tau0", tau0), kw.get("depth", depth), cache, max_depth)
    if depth > 0 and len(flips):  # neurons that flip at the entry point too: apply them before judging the gaps
        pending = ~_denoise(flips - tau0, np.maximum(np.linalg.norm(flips, axis=1), np.linalg.norm(tau0))).any(1)
        if pending.any():  # earliest step first: toggling a neuron changes only later steps' rates
            new = masks.copy()
            tt, i = divmod(int(moving[np.argmax(pending)]), rnn.d)
            new[tt, i] = ~new[tt, i]
            return again(Q, masks=new, depth=depth + 1)
    others = np.array([j for j in range(len(a)) if j != t])
    F0 = a[others] + b[others, None] * tau0  # every gap at the entry point, as a functional
    F0 = _denoise(F0, np.linalg.norm(a[others], axis=1) + np.abs(b[others]) * np.linalg.norm(tau0))
    nrm = np.linalg.norm(F0, axis=1)
    v = (Q.V @ F0.T) / np.where(nrm > 0, nrm, 1.0)
    pos, negv = (v > TOL).any(0), (v < -TOL).any(0)
    fl.split += 3 * len(Q.V) * m * len(others)
    mixed = pos & negv
    if mixed.any():  # the sign of every gap at entry must be the same on all of Q
        return sum(again(K) for K in split(Q, F0[np.argmax(mixed)], fl))
    bscale = np.abs(b).max() + 1e-300
    zero = ~pos & ~negv  # the gap vanishes at entry on all of Q: its sign just after entry is its slope's
    st = np.where(negv, -1, 1)
    st = np.where(zero & (b[others] < -1e-12 * bscale), -1, st)
    t_first = others > t  # on an exact tie the lower index is predicted
    flat = tied[others] | (zero & (np.abs(b[others]) <= 1e-12 * bscale))  # tied all along the line
    st = np.where(flat, np.where(t_first, 1, -1), st)
    neg = others[st < 0]
    if len(neg) == 0:  # t is the argmax at entry: no sliver starts here (or it ended at the crossing)
        return 0.0 if depth == 0 else acc.add(Q, tau0)
    lo = None
    lower_ok = ~flat[st < 0] & (b[neg] > 0)
    if lower_ok.all():  # otherwise t never wins in this region
        lo = _extreme(Q, -a[neg] / b[neg, None], fl, max)  # t wins once tau >= every lower root ...
        if isinstance(lo, tuple):
            return sum(again(K) for K in lo)
        up = (st > 0) & ~flat & ~zero & (b[others] < 0)
        upper = others[up]  # ... and while tau <= every upper root
        if len(upper):
            hi = _extreme(Q, -a[upper] / b[upper, None], fl, min)
            if isinstance(hi, tuple):
                return sum(again(K) for K in hi)
            gap = _denoise(hi - lo, max(np.linalg.norm(hi), np.linalg.norm(lo)))
            sg = _signs(Q, gap[None])[0]
            if sg == 0:
                return sum(again(K) for K in split(Q, gap, fl))
            if sg < 0 or not gap.any():  # t never wins, or only at single points of the line
                lo = None
    V = Q.V
    caps_arr = np.array(caps).reshape(-1, m)
    # neuron flips that can come before every end: a flip >= a cap on all of Q is not, nor a flip
    # >= the target's win on all of Q unless it coincides with it (then the flip is taken first)
    keep = np.ones(len(flips), bool)
    if len(flips):
        for ends, strict in ((caps_arr, False), (None if lo is None else lo[None], True)):
            if ends is None or not len(ends):
                continue
            diff = _denoise(flips[:, None, :] - ends[None, :, :],  # (k, e, m)
                            np.maximum(np.linalg.norm(flips, axis=1)[:, None], np.linalg.norm(ends, axis=1)[None]))
            dn = np.linalg.norm(diff, axis=-1)
            vals = np.einsum("rm,kem->rke", V, diff) / np.where(dn > 0, dn, 1.0)
            after = (vals >= -TOL).all(0)
            if strict:
                after &= (vals > TOL).any(0)
            keep &= ~after.any(1)
    cand, cand_id = flips[keep], moving[keep]
    fl.split += 2 * len(V) * m * len(flips) * (len(caps_arr) + 1)
    # events in tie-breaking order: caps, flips, then the target's win
    events = [caps_arr, cand] + ([lo[None]] if lo is not None else [])
    kinds = ["cap"] * len(caps_arr) + ["flip"] * len(cand) + (["win"] if lo is not None else [])
    events = np.concatenate(events)
    if not len(events):
        raise RuntimeError("unbounded sliver (no cap, no flip and the target never wins)")
    centre = V.sum(0)
    size = np.linalg.norm(events, axis=1)
    k = int(np.argmin(events @ centre))
    same = ~_denoise(events - events[k], np.maximum(size, size[k])).any(1)
    k = int(np.argmax(same))  # of events equal to the earliest one, the first in tie-breaking order
    for _ in range(len(events)):  # the first event must be the same on all of Q
        diff = _denoise(events - events[k], np.maximum(size, size[k]))
        dn = np.linalg.norm(diff, axis=1)
        vals = (V @ diff.T) / np.where(dn > 0, dn, 1.0)
        low = vals.min(0)
        low[k] = 0.0
        if not (low < -TOL).any():
            break
        j = int(np.argmin(low))  # split along the event that undercuts the current one the most
        plus, minus = split(Q, diff[j], fl)
        if plus is not None and minus is not None:
            return sum(again(K) for K in (plus, minus))
        k = j  # it is <= the current one on all of Q: it is the new candidate
    else:
        raise RuntimeError("no uniform first event")
    if kinds[k] != "flip":
        return acc.add(Q, events[k])
    new = masks.copy()
    tt, i = divmod(int(cand_id[k - len(caps_arr)]), rnn.d)
    new[tt, i] = ~new[tt, i]
    return again(Q, masks=new, tau0=events[k], depth=depth + 1)


def _extreme(Q, funcs, fl, pick):
    """The functional among the rows of funcs that is the largest (pick=max) or smallest
    (pick=min) on all of Q, or, if that differs across Q, the two halves of Q split where two of
    them cross."""
    centre = Q.V.sum(0)
    vals = funcs @ centre
    k = int(np.argmax(vals) if pick is max else np.argmin(vals))
    size = np.linalg.norm(funcs, axis=1)
    diff = _denoise(funcs - funcs[k], np.maximum(size, size[k]))
    sg = _signs(Q, diff)
    sg[k] = 1
    if (sg == 0).any():
        return split(Q, diff[np.argmax(sg == 0)], fl)
    return funcs[k]


def sliver_error(rnn, task, return_details=False, order_cap=True, pairs=None, cap=None, width="first"):
    """Tie-sliver estimate of the error rate (1 - accuracy). Returns (error, Flops, details) with
    details = dict(flagged H-measure where the first-order picture fails). With `pairs` (a set
    of (p, q), p < q) only those tie hyperplanes are included (the sum over all pairs is the
    estimate; sliver_table.py uses this to spread one model over several processes).

    width="first" (the first-order laws): the width is -Delta / g on the part of H_pq where the
    competitor c is the argmax at the tie, with Delta the gap l_t - l_c there and g its growth
    rate moving off H_pq, and the measure is phi(0) E[width].
    width="exact" (the exact sliver law): the width is the true length of the wrong stretch along
    each normal line, followed exactly through the linear regions it crosses (_walk), and the
    measure is the exact Gaussian measure of the sliver.

    cap="none" (the first-order law with width="first"), "chamber" (the order-truncated law: the
    width is capped where the normal line leaves the order chamber) or "bisector" (each wrong
    input belongs to the nearer of its target's two walls, so the sliver of a wall stops at the
    bisector between them, a third of the way to the other wall, and at the chamber's other
    exits). With the bisector cap the slivers of different walls are disjoint, so the exact sliver
    law is the measure of a set of wrong inputs: the exact error minus the wrong inputs that are
    not on a wrong stretch from their nearer wall. Without cap, the old argument order_cap
    selects "chamber" (True) or "none" (False).

    Near a triple tie the first-order width can be O(1) even for a well-trained model (a large
    logit gap at the tie, with a third input just above or below), and without a cap those points
    dominate the 5-input estimates."""
    if cap is None:
        cap = "chamber" if order_cap else "none"
    fl = Flops()
    err = 0.0
    flagged = 0.0
    n = rnn.n
    acc = _Slivers(fl)
    for p, q, E, nu, roots, limits, targets in tie_frames(n, task):
        if pairs is not None and (p, q) not in pairs:
            continue
        for R, lim in zip(roots, limits):
            caps = _caps(lim, cap)
            for s in (+1, -1):
                t, c = targets[s]
                for cone, L, masks in tie_pieces(rnn, Cone.simplicial(R), E, s * nu, fl):
                    if width == "exact":
                        _walk(rnn, cone, masks, E, s * nu, t, caps, fl, acc)
                        continue
                    dvec = L[t] - L[c]
                    delta = E.T @ dvec  # gap at the tie as a function of zeta
                    g = float(dvec @ (s * nu))  # growth of the gap moving into side s
                    _, wrong = split(cone, delta, fl)  # where the model prefers c at the tie
                    if wrong is None:
                        continue
                    # within it, c must be the argmax among all classes at the tie
                    K = wrong
                    for rr in range(len(L)):
                        if rr in (t, c) or K is None:
                            continue
                        K, other = split(K, E.T @ (L[c] - L[rr]), fl)
                    if K is None:
                        continue
                    if g <= 0:
                        flagged += 1.0  # count of non-thin pieces (diagnostic)
                        continue
                    if caps:
                        err += PHI0 * min_moment(K, [-delta / g] + caps, fl)
                    else:
                        err += PHI0 * linear_moment(K, -delta / g, fl)
    err += acc.flush()
    details = dict(flagged=flagged)
    return (err, fl, details) if return_details else (err, fl)
