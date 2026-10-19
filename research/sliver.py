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

from engine import TOL, Cone, Flops, _maximal, _pull, split, target_rank
from orthant import cone_measure

PHI0 = 1.0 / math.sqrt(2 * math.pi)


def tie_frames(n, task):
    """Yields (p, q, E, nu, roots, targets): E (n, n-1) orthonormal basis of H_pq whose first
    column is (e_p + e_q)/sqrt 2, nu = (e_p - e_q)/sqrt 2, roots = simplicial cones (rays in
    zeta coordinates) covering the part of H_pq where the tie is between the target rank and an
    adjacent rank, and targets = {side: (target, competitor)} for side in (+1, -1)."""
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
            roots = []
            for perm in itertools.permutations(range(m)):
                # perm[i] = object at sorted rank i; object 0 is the tie, object j>0 = others[j-1]
                if perm[obj_rank] != 0:
                    continue
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
            yield p, q, E, nu, roots, targets


def tie_pieces(rnn, cone, E, side_dir, fl):
    """Linear pieces of the RNN on a cone in H (zeta coordinates), taking for each neuron whose
    pre-activation vanishes identically on the piece the sign it has just off H in direction
    side_dir (the one-sided limit). Yields (cone, L) with L the logits map in x coordinates."""
    d, n = rnn.d, rnn.n
    stack = [(cone, 0, np.zeros((d, n)))]
    while stack:
        cone, t, H = stack.pop()
        split_done = False
        Vx = cone.V @ E.T
        while t < n:
            P = rnn.Wh @ H
            P[:, t] += rnn.Wi
            S = Vx @ P.T
            sc = np.linalg.norm(P, axis=1) + 1e-300
            pos = S > TOL * sc
            neg = S < -TOL * sc
            fl.rnn += 2 * d * d * t + 2 * len(Vx) * (t + 1) * d + 6 * len(Vx) * d
            conflict = pos.any(0) & neg.any(0)
            if conflict.any():
                i = np.where(conflict)[0][0]
                plus, minus = split(cone, E.T @ P[i], fl)
                stack.append((plus, t, H))
                stack.append((minus, t, H))
                split_done = True
                break
            zero = ~pos.any(0) & ~neg.any(0)
            mask = pos.any(0) | (zero & (P @ side_dir > TOL * sc))
            H = mask[:, None] * P
            t += 1
        if not split_done:
            fl.rnn += 2 * len(rnn.Wo) * d * n
            yield cone, rnn.Wo @ H


def facet_measures(cone, fl):
    """(unit inward normals (k, m), (m-1)-dim Gaussian measures (k,)) of the facets of a cone in
    R^m, the latter computed in each facet's own hyperplane."""
    m = cone.V.shape[1]
    normals, meas = [], []
    facets = [f for f in range(cone.I.shape[1]) if cone.I[:, f].sum() >= m - 1]
    all_sets = [frozenset(np.where(cone.I[:, f])[0].tolist()) for f in facets]
    for f, S in zip(facets, all_sets):
        nrm = cone.F[f]
        if m == 1:  # a ray: its only facet is the origin, of (0-dim) measure 1
            normals.append(nrm)
            meas.append(1.0)
            continue
        if m == 2:
            normals.append(nrm)
            meas.append(0.5)
            continue
        # orthonormal basis of the facet hyperplane
        Q = np.linalg.svd(np.eye(m) - np.outer(nrm, nrm))[0][:, : m - 1]
        if len(S) == m - 1:
            simps = [tuple(sorted(S))]
        else:
            ridges = _maximal(S & T for T in all_sets if T is not S and len(S & T) >= m - 2)
            simps = _pull(sorted(S), ridges, m - 1)
        R = cone.V[np.array(simps)] @ Q  # (s, m-1, m-1)
        normals.append(nrm)
        meas.append(cone_measure(R).sum())
        fl.measure += len(simps) * (60 + 4 * m**3)
    return np.array(normals).reshape(-1, m), np.array(meas)


def first_moment(cone, fl):
    """E[zeta 1_K(zeta)] for zeta ~ N(0, I_m), via Gaussian integration by parts."""
    N, g = facet_measures(cone, fl)
    return PHI0 * (N * g[:, None]).sum(0)


def sliver_error(rnn, task, return_details=False):
    """First-order tie-sliver estimate of the error rate (1 - accuracy). Returns (error, Flops,
    details) with details = dict(flagged H-measure where the first-order picture fails)."""
    fl = Flops()
    err = 0.0
    flagged = 0.0
    n = rnn.n
    for p, q, E, nu, roots, targets in tie_frames(n, task):
        for R in roots:
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
                    mu = first_moment(K, fl)
                    err += PHI0 / g * float(-delta @ mu)
    details = dict(flagged=flagged)
    return (err, fl, details) if return_details else (err, fl)
