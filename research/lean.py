"""A second finishing rule and ladder for 3-input models ("lean"), designed on development seeds
5000-5131 after the held-out test of the lguard ladder (see README, "A second ladder for 3-input
models"). It changes three things relative to engine.anytime_ladder with rule="lguard":

* Exact finishing of implausible last-step conflicts. A cone whose first conflict is at the last
  step and whose tie linearisation claims more than tau of it is wrong is finished exactly: at
  the last step every piece of an exact split is a linear region. (These cones carried the
  mid-range failure of lguard: thin tie slivers whose linear map the linearisation extended over
  the whole cone.) Earlier conflicts keep lguard's central-ray guard for cones of measure >= big.
* Two-face linearisation. A cone with both of its target's tie faces (the walls to the ranks
  above and below) is split along the plane equidistant from them (the bisector of the exact
  sliver law), and each half is linearised next to its own face, so that each wall's sliver gets
  that wall's activation pattern. engine._tie_point uses the first face only.
* The roots_k rungs examine root cones in decreasing order of their closed-form Gaussian measure
  (the order depends only on the geometry, not on the weights).

and three pieces of bookkeeping that give the same estimates with fewer charged operations:

* measure3: the Gaussian measure of a pointed cone in R^3 by Girard's theorem from the angle
  between the two facet normals at each extreme ray, gamma = (2 pi - sum of angles) / (4 pi),
  charged 45 flops per ray (cross product, norm, dot, atan2 at 20) plus an incidence scan,
  instead of triangulating and measuring simplices with arcsines;
* resolve_lean: a linearisation at a point needs only the point's pre-activations at later steps
  (for a neuron on which the rays agree, the point's sign is theirs), not the rays';
* advance_lean: at step 0 of a root every pre-activation is w_i x_0 and the sign of x_0 is fixed
  on the root, so the first activation pattern is known without evaluating rays.

The exact rung is exact refinement with the same bookkeeping (exact_accuracy_lean); it agrees with
engine.exact_accuracy to about 1e-10.
"""

import heapq
import itertools
import math

import numpy as np

from engine import (Cone, Flops, _heap_flops, _preact_map, _ray_preacts, _root_setup_flops, _signs,
                    _split_children, _tie_point, cut_incorrect, measure_simplices, root_cones, split,
                    tie_normals, triangulate)

FOURPI = 4 * math.pi


def measure3(cone, fl):
    """Gaussian measure of a pointed polyhedral cone in R^3 (Girard's theorem). Falls back to
    triangulation if the cone is not 3-dimensional or a ray is not on exactly two facets."""
    V, F, I = cone.V, cone.F, cone.I
    m, k = I.shape
    fl.split += m * k  # incidence scan
    if V.shape[1] != 3 or (I.sum(1) != 2).any():
        simps = triangulate(cone)
        fl.split += 5 * len(V) * V.shape[1]
        return float(measure_simplices(simps, V.shape[1], fl).sum())
    idx = np.argsort(~I, axis=1, kind="stable")[:, :2]
    a, b = F[idx[:, 0]], F[idx[:, 1]]
    ang = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), (a * b).sum(1))
    fl.measure += 45 * m + 3
    return float(min(max((2 * math.pi - ang.sum()) / FOURPI, 0.0), 0.5))


def incorrect_lean(rnn, cone, L, b, fl):
    """Measure of the part of a cone where the logits map L does not predict b."""
    return sum(measure3(p, fl) for p in cut_incorrect(cone, L, b, fl))


def region_incorrect_lean(rnn, cone, H, b, fl):
    """Exact incorrect measure of a cone on which the RNN is linear with hidden map H."""
    fl.rnn += 2 * len(rnn.Wo) * rnn.d * rnn.n
    return incorrect_lean(rnn, cone, rnn.Wo @ H, b, fl)


def resolve_lean(rnn, t, H, P, point, fl):
    """engine._resolve_rest from step t (pre-activation map P): every neuron takes its sign at
    `point` (for neurons on which the cone's rays agree this is their common sign). Returns the
    logits map."""
    d, n = rnn.d, rnn.n
    while True:
        at = point[: t + 1] @ P[:, : t + 1].T > 0
        fl.rnn += 2 * (t + 1) * d + d * (t + 1)
        H = at[:, None] * P
        t += 1
        if t == n:
            break
        P = rnn.Wh @ H
        P[:, t] += rnn.Wi
        fl.rnn += 2 * d * d * t + d
    fl.rnn += 2 * len(rnn.Wo) * d * n
    return rnn.Wo @ H


def advance_lean(rnn, cone, t, H, P, S, fl):
    """engine._advance_to_conflict, with step 0 of a root resolved from the sign of x_0."""
    d, n = rnn.d, rnn.n
    if t == 0 and P is None:
        s0 = 1.0 if cone.V[:, 0].sum() > 0 else -1.0
        P = np.zeros((d, n))
        P[:, 0] = rnn.Wi
        H = ((rnn.Wi * s0) > 0)[:, None] * P
        fl.rnn += 2 * d
        t = 1
        P = S = None
    while t < n:
        if P is None:
            P = _preact_map(rnn, H, t, fl)
            S = _ray_preacts(cone.V, P, t, fl)
        pos, neg = _signs(S, P, t, fl)
        conflict = pos.any(0) & neg.any(0)
        if conflict.any():
            return "conflict", t, H, P, S, int(np.where(conflict)[0][0])
        H = pos.any(0)[:, None] * P
        fl.rnn += d * (t + 1)
        t += 1
        P = S = None
    return "done", H


def exact_lean(rnn, cone, t, H, P, S, b, fl):
    """Exact incorrect measure of a cone from refinement state (t, H, P, S)."""
    stack = [(cone, t, H, P, S)]
    e = 0.0
    while stack:
        cone, t, H, P, S = stack.pop()
        item = advance_lean(rnn, cone, t, H, P, S, fl)
        if item[0] == "done":
            e += region_incorrect_lean(rnn, cone, item[1], b, fl)
            continue
        _, t2, H2, P2, S2, i = item
        for child, Sc in _split_children(cone, P2, S2, i, t2, fl):
            stack.append((child, t2, H2, P2, Sc))
    return e


def exact_accuracy_lean(rnn, task):
    """Exact accuracy by refinement with the lean bookkeeping. Returns (accuracy, Flops)."""
    fl = Flops()
    e = 0.0
    for R, b, _ in root_cones(rnn.n, task):
        fl.split += _root_setup_flops(rnn.n)
        e += exact_lean(rnn, Cone.simplicial(R), 0, np.zeros((rnn.d, rnn.n)), None, None, b, fl)
    return 1.0 - e, fl


def _tie_faces(cone, ties):
    n = cone.V.shape[1]
    return [c for c in ties if (np.abs(cone.V @ c) <= 1e-9).sum() >= n - 1]


def finish_lean(rnn, cone, t, H, P, S, b, ties, fl, tau=0.1, big=0.01):
    """Incorrect measure of a cone that will not be split by the ladder (see the module
    docstring): exact if it is a single linear region; else linearised next to its tie face, or
    next to each of its two tie faces on the two sides of their bisector; a linearisation that
    claims more than tau of the cone is wrong is replaced by the exact value if the first
    conflict is at the last step, and otherwise checked against the central ray for cones of
    measure >= big (engine's lguard)."""
    d, n = rnn.d, rnn.n
    item = advance_lean(rnn, cone, t, H, P, S, fl)
    if item[0] == "done":
        return region_incorrect_lean(rnn, cone, item[1], b, fl)
    _, t, H, P, S, i = item
    fl.rnn += 2 * len(cone.V) * n * len(ties)  # which rays lie on which tie face
    faces = _tie_faces(cone, ties)
    if len(faces) == 2:
        plus, minus = split(cone, faces[1] - faces[0], fl)  # plus: the half nearer faces[0]
        if plus is None or minus is None:
            fl.rnn += 2 * len(cone.V) * n
            e_tie = incorrect_lean(rnn, cone, resolve_lean(rnn, t, H, P, _tie_point(cone, faces[:1]), fl), b, fl)
        else:
            e_tie = 0.0
            for half, face in ((plus, faces[0]), (minus, faces[1])):
                fl.rnn += 2 * len(half.V) * n * 2  # its face rays and the tie point
                pt = _tie_point(half, [face])
                e_tie += incorrect_lean(rnn, half, resolve_lean(rnn, t, H, P, pt, fl), b, fl)
    else:
        fl.rnn += 2 * len(cone.V) * n
        e_tie = incorrect_lean(rnn, cone, resolve_lean(rnn, t, H, P, _tie_point(cone, faces), fl), b, fl)
    if e_tie == 0.0:
        return e_tie
    mu = measure3(cone, fl)
    if e_tie <= tau * mu:
        return e_tie
    if t == n - 1:  # implausible at the last step: finish exactly
        return exact_lean(rnn, cone, t, H, P, S, b, fl)
    if mu < big:
        return e_tie
    fl.rnn += 2 * len(cone.V) * n
    e_ctr = incorrect_lean(rnn, cone, resolve_lean(rnn, t, H, P, cone.V.sum(0), fl), b, fl)
    return min(e_tie, e_ctr) if abs(e_tie - e_ctr) > tau * mu else e_tie


def anytime_ladder_lean(rnn, task, split_grid, root_grid=(), **kw):
    """engine.anytime_ladder with the lean finishing rule and bookkeeping, and the roots_k rungs
    in decreasing order of root measure. Returns a dict name -> (accuracy estimate, flops)."""
    d, n = rnn.d, rnn.n
    out = {}
    roots = sorted(root_cones(n, task), key=lambda r: -r[2])  # largest first (stable)
    cum = []
    for k, (R, b, _) in enumerate(roots):
        f = Flops()
        cone = Cone.simplicial(R)
        f.split += _root_setup_flops(n) + _heap_flops(k) + 2 * len(cone.V) ** 2 * n
        f.split += _heap_flops(len(roots) - k)  # its pop (heap order does not change the sum)
        e = finish_lean(rnn, cone, 0, np.zeros((d, n)), None, None, b, tie_normals(R, task), f, **kw)
        cum.append((e, f.total))
    for k in sorted(set(root_grid)):
        if k < len(roots):
            out[f"roots{k}"] = (1.0 - sum(e for e, _ in cum[:k]), sum(c for _, c in cum[:k]))
    grid = sorted(set(split_grid))
    fl = Flops()
    heap = []
    counter = itertools.count()

    def width(cone):
        fl.split += 2 * len(cone.V) ** 2 * n
        return float((cone.V @ cone.V.T).min())  # min cosine between rays: smaller = wider

    for R, b, _ in roots:
        cone = Cone.simplicial(R)
        fl.split += _root_setup_flops(n) + _heap_flops(len(heap))
        heapq.heappush(heap, (width(cone), next(counter), cone, 0, np.zeros((d, n)), None, None, b,
                              tie_normals(R, task)))
    splits, err = 0, 0.0

    def snapshot():
        f = Flops()
        e = err
        size = len(heap)
        for item in heap:
            f.split += _heap_flops(size)
            size -= 1
            _, _, cone, t, H, P, S, b, ties = item
            e += finish_lean(rnn, cone, t, H, P, S, b, ties, f, **kw)
        return 1.0 - e, fl.total + f.total

    gi = 0
    while gi < len(grid) and grid[gi] <= splits:
        out[f"any{grid[gi]}"] = snapshot()
        gi += 1
    while heap and gi < len(grid):
        fl.split += _heap_flops(len(heap))
        _, _, cone, t, H, P, S, b, ties = heapq.heappop(heap)
        item = advance_lean(rnn, cone, t, H, P, S, fl)
        if item[0] == "done":
            err += region_incorrect_lean(rnn, cone, item[1], b, fl)
            continue
        _, t2, H2, P2, S2, i = item
        for child, S_child in _split_children(cone, P2, S2, i, t2, fl):
            fl.split += _heap_flops(len(heap))
            heapq.heappush(heap, (width(child), next(counter), child, t2, H2, P2, S_child, b, ties))
        splits += 1
        while gi < len(grid) and grid[gi] <= splits:
            out[f"any{grid[gi]}"] = snapshot()
            gi += 1
    if gi < len(grid):  # refinement finished early: remaining rungs are exact
        while heap:
            fl.split += _heap_flops(len(heap))
            _, _, cone, t, H, P, S, b, ties = heapq.heappop(heap)
            err += finish_lean(rnn, cone, t, H, P, S, b, ties, fl, **kw)
        final = (1.0 - err, fl.total)
        for k in grid[gi:]:
            out[f"any{k}"] = final
    return out
