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
from slice import slice_flops, slice_linear_moment

PHI0 = 1.0 / math.sqrt(2 * math.pi)


def tie_frames(n, task):
    """Yields (p, q, E, nu, roots, limits, targets): E (n, n-1) orthonormal basis of H_pq whose
    first column is (e_p + e_q)/sqrt 2, nu = (e_p - e_q)/sqrt 2, roots = simplicial cones (rays
    in zeta coordinates) covering the part of H_pq where the tie is between the target rank and
    an adjacent rank, limits[i] = the order limits of roots[i] (see below), and
    targets = {side: (target, competitor)} for side in (+1, -1).

    Order limits. Moving a distance tau off H_pq along +-nu, one tied value rises and the other
    falls by tau / sqrt 2, so the line leaves the order chamber of the tie face (the ordering
    changes) once the riser passes the object just above the tie, at
    tau = sqrt 2 x_above - zeta_0, or the faller passes the one just below, at
    tau = zeta_0 - sqrt 2 x_below. Within a root the ordering is fixed, so both are linear
    functionals of zeta (nonnegative on the root); limits holds those that exist."""
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
                lim = []
                if obj_rank >= 1:  # sqrt 2 zeta_above - zeta_0
                    lim.append(math.sqrt(2) * np.eye(m)[perm[obj_rank - 1]] - np.eye(m)[0])
                if obj_rank + 1 <= m - 1:  # zeta_0 - sqrt 2 zeta_below
                    lim.append(np.eye(m)[0] - math.sqrt(2) * np.eye(m)[perm[obj_rank + 1]])
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
    side_dir (the one-sided limit). Yields (cone, L) with L the logits map in x coordinates."""
    d, n = rnn.d, rnn.n
    m = E.shape[1]
    # stack items: (cone, t, H, (P, PE, sc) or None, S); as in engine.linear_pieces, children
    # inherit the step-t pre-activation map and their parent's ray pre-activations, so only
    # the new rays of a split are evaluated
    stack = [(cone, 0, np.zeros((d, n)), None, None)]
    while stack:
        cone, t, H, maps, S = stack.pop()
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
                stack.append((plus, t, H, maps, np.concatenate([S[Pi], S[Zi], SW])))
                stack.append((minus, t, H, maps, np.concatenate([S[Ni], S[Zi], SW])))
                split_done = True
                break
            zero = ~pos.any(0) & ~neg.any(0)
            mask = pos.any(0) | (zero & (P @ side_dir > TOL * sc))
            H = mask[:, None] * P
            fl.rnn += 2 * d * n + d * (t + 1)
            t += 1
            maps = S = None
        if not split_done:
            fl.rnn += 2 * len(rnn.Wo) * d * n
            yield cone, rnn.Wo @ H


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


def sliver_error(rnn, task, return_details=False, order_cap=True, pairs=None):
    """Tie-sliver estimate of the error rate (1 - accuracy). Returns (error, Flops, details) with
    details = dict(flagged H-measure where the first-order picture fails). With `pairs` (a set
    of (p, q), p < q) only those tie hyperplanes are included (the sum over all pairs is the
    estimate; sliver_table.py uses this to spread one model over several processes).

    order_cap=False is the first-order law: width -Delta / g everywhere on the wrong part of
    H_pq. order_cap=True (the order-truncated law) caps the width at the order limits of the
    root (see tie_frames), i.e. where the normal line leaves the order chamber. Within a chamber
    the target is fixed and the model's errors are slivers on the target's two walls, so each
    sliver is measured inside its chamber. Near a triple tie the first-order width can be O(1)
    even for a well-trained model (a large logit gap at the tie, with a third input just above
    or below), and without the cap those points dominate the 5-input estimates. (Capping only
    where the target itself changes, i.e. ignoring the competitor's crossing, differs at second
    order: +20% instead of +18% for M_{16,5}, by Monte Carlo.)"""
    fl = Flops()
    err = 0.0
    flagged = 0.0
    n = rnn.n
    for p, q, E, nu, roots, limits, targets in tie_frames(n, task):
        if pairs is not None and (p, q) not in pairs:
            continue
        for R, lim in zip(roots, limits):
            for s in (+1, -1):
                t, c = targets[s]
                for cone, L in tie_pieces(rnn, Cone.simplicial(R), E, s * nu, fl):
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
                    if order_cap and lim:
                        err += PHI0 * min_moment(K, [-delta / g] + lim, fl)
                    else:
                        err += PHI0 * linear_moment(K, -delta / g, fl)
    details = dict(flagged=flagged)
    return (err, fl, details) if return_details else (err, fl)
