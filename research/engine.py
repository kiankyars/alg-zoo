"""Deterministic (sampling-free) accuracy of bias-free ReLU RNN classifiers on Gaussian inputs.

Key facts used:
  * No biases => the RNN is positively homogeneous and piecewise linear, so correctness only
    depends on the direction of x, and every linear region is a polyhedral cone.
  * The target (2nd argmax / argmedian) only depends on the ordering of x. Splitting R^n by
    ordering and by the sign of x_0 gives 2 n! simplicial "root" cones with a fixed target and a
    fixed first activation pattern, with Gaussian measures known in closed form.
  * A cone whose extreme rays all agree on the sign of a pre-activation lies on one side of
    that neuron's hyperplane (the region is convex). Processing time steps in order, the
    pre-activations at step t are linear on any cone not split before t. Where rays disagree
    the cone is split exactly along that hyperplane (double description method).
  * Within a single linear region the logits are linear, so the incorrect part is cut out by
    at most n_out - 1 more hyperplanes. Gaussian measures of the resulting cones are computed
    by triangulation plus closed forms / Plackett's formula (orthant.py) or slice cubature
    (slice.py).

exact_accuracy gives the accuracy up to quadrature / floating point error; anytime_accuracy is
a cheaper estimator with a compute knob. Neither uses random or pseudo-random inputs.
"""

import heapq
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from orthant import cone_measure, orthant_flops
from slice import slice_flops, slice_measure

TOL = 1e-10


@dataclass
class RNN:
    Wi: np.ndarray  # (d,)
    Wh: np.ndarray  # (d, d)
    Wo: np.ndarray  # (n_out, d)
    n: int  # sequence length

    @classmethod
    def from_torch(cls, model):
        return cls(
            Wi=model.rnn.weight_ih_l0.detach().double().numpy()[:, 0].copy(),
            Wh=model.rnn.weight_hh_l0.detach().double().numpy().copy(),
            Wo=model.linear.weight.detach().double().numpy().copy(),
            n=model.seq_len,
        )

    @property
    def d(self):
        return len(self.Wi)

    def forward(self, X):
        h = np.zeros(X.shape[:-1] + (self.d,))
        for t in range(self.n):
            h = np.maximum(h @ self.Wh.T + X[..., t : t + 1] * self.Wi, 0.0)
        return h @ self.Wo.T

    def forward_flops(self):
        """Cost of one forward pass, the unit charged to sampling per sample. Multiply-adds count
        as 2 flops. h_0 = 0, so step 0 is just w x_0 (the engine does not pay for W_hh h_0
        either); random number generation and computing the target are not charged."""
        d, n = self.d, self.n
        return 2 * d + (n - 1) * (2 * d * d + 2 * d) + 2 * len(self.Wo) * d


def target_rank(task, n):
    """Rank (0 = largest) of the target position for order-based classification tasks.
    alg_zoo's argmedian is argsort (ascending) index n // 2, i.e. rank n - 1 - n // 2 from the top."""
    return {"2nd_argmax": 1, "argmax": 0, "argmedian": n - 1 - n // 2}[task]


def root_cones(n, task):
    """Yields (rays (n, n), target position, exact Gaussian measure) for the 2 n! simplicial
    cones {x_{p1} >= ... >= x_{pn}, +-x_0 >= 0} (p = ordering by value), on each of which the
    target is fixed and the first input has a fixed sign, so the RNN's first activation pattern
    (every pre-activation at step 0 is w_i x_0) is constant. If x_0 has rank r in the ordering,
    the cone with x_0 >= 0 has measure P(at least r of the n inputs are positive) / n!, and the
    one with x_0 <= 0 the complement. Rays (in sorted coordinates): +-(1, ..., 1) and, for
    0 < j < n, (1^j, 0^(n-j)) if j < r else (0^j, (-1)^(n-j))."""
    tr = target_rank(task, n)
    for perm in itertools.permutations(range(n)):
        r = perm.index(0) + 1  # 1-based rank of x_0
        blocks = []
        for j in range(1, n):
            y = np.zeros(n)
            if j < r:
                y[:j] = 1.0
            else:
                y[j:] = -1.0
            blocks.append(y)
        p_pos = sum(math.comb(n, k) for k in range(r, n + 1)) / 2**n
        for sgn, meas in ((1.0, p_pos), (-1.0, 1.0 - p_pos)):
            Y = np.stack([sgn * np.ones(n)] + blocks)
            X = np.zeros((n, n))
            X[:, list(perm)] = Y
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            yield X, perm[tr], meas / math.factorial(n)


@dataclass
class Flops:
    rnn: float = 0.0
    split: float = 0.0
    measure: float = 0.0
    counts: dict = field(default_factory=lambda: dict(splits=0, leaves=0, simplices=0))

    @property
    def total(self):
        return self.rnn + self.split + self.measure

    def add(self, other):
        self.rnn += other.rnn
        self.split += other.split
        self.measure += other.measure
        for k, v in other.counts.items():
            self.counts[k] = self.counts.get(k, 0) + v


class Cone:
    """Pointed polyhedral cone {x : F x >= 0} with extreme rays V and incidence I[ray, facet]."""

    __slots__ = ("V", "F", "I")

    def __init__(self, V, F, I):
        self.V, self.F, self.I = V, F, I

    @classmethod
    def simplicial(cls, R):
        n = len(R)
        idx = np.arange(n)
        cof = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                cof[i, j] = (-1) ** (i + j) * np.linalg.det(R[idx != i][:, idx != j])
        F = cof / np.linalg.norm(cof, axis=1, keepdims=True)
        F *= np.sign((F * R).sum(1))[:, None]
        return cls(R.copy(), F, ~np.eye(n, dtype=bool))


def split(cone, c, fl=None, parts=False):
    """Exact split of a cone by the hyperplane c.x = 0. Returns (plus, minus); either may be
    None if the cone lies entirely on one side. With parts=True also returns
    (P, N, Z, W): indices of the parent's rays strictly on the + / - side / on the hyperplane,
    and the new rays W; the children's rays are [V[P], V[Z], W] and [V[N], V[Z], W]."""
    V, F, I = cone.V, cone.F, cone.I
    m, n = V.shape
    c = c / np.linalg.norm(c)
    s = V @ c
    P = np.where(s > TOL)[0]
    N = np.where(s < -TOL)[0]
    if len(P) == 0 or len(N) == 0:
        out = (None, cone) if len(P) == 0 else (cone, None)
        return out + (None,) if parts else out
    Z = np.where(np.abs(s) <= TOL)[0]
    k = I.shape[1]
    K = I[P][:, None, :] & I[N][None, :, :]  # common facets of each (p, q) pair
    cand = np.argwhere(K.sum(-1) >= n - 2)
    Kc = K[cand[:, 0], cand[:, 1]]  # (M, k)
    # adjacency: no third extreme ray lies on all common facets
    contains = (I[None, :, :] | ~Kc[:, None, :]).all(-1).sum(-1)
    adj = cand[contains == 2]
    Kc = Kc[contains == 2]
    p, q = P[adj[:, 0]], N[adj[:, 1]]
    W = s[p][:, None] * V[q] - s[q][:, None] * V[p]
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    if fl is not None:
        fl.split += 2 * m * n + len(P) * len(N) * k + len(cand) * m * k + 6 * len(adj) * n
        fl.counts["splits"] += 1
    out = []
    for side, sign in ((P, 1.0), (N, -1.0)):
        Vs = np.concatenate([V[side], V[Z], W])
        Is = np.concatenate(
            [
                np.concatenate([I[side], np.zeros((len(side), 1), bool)], 1),
                np.concatenate([I[Z], np.ones((len(Z), 1), bool)], 1),
                np.concatenate([Kc, np.ones((len(Kc), 1), bool)], 1),
            ]
        )
        Fs = np.concatenate([F, sign * c[None]])
        keep = Is.sum(0) >= n - 1  # drop facets that are no longer facets
        out.append(Cone(Vs, Fs[keep], Is[:, keep]))
    if parts:
        return out[0], out[1], (P, N, Z, W)
    return out[0], out[1]


def _maximal(sets):
    sets = sorted(set(sets), key=len, reverse=True)
    out = []
    for s in sets:
        if not any(s <= t for t in out):
            out.append(s)
    return out


def _pull(rays, facets, dim):
    if len(rays) == dim:
        return [tuple(rays)]
    v = rays[0]
    out = []
    for S in facets:
        if v in S:
            continue
        ridges = _maximal(S & T for T in facets if T is not S and len(S & T) >= dim - 2)
        for simp in _pull(sorted(S), ridges, dim - 1):
            out.append((v,) + simp)
    return out


def triangulate(cone):
    """Pulling triangulation of a pointed cone; returns array (n_simplices, n, n) of rays."""
    m, n = cone.V.shape
    if m == n:
        return cone.V[None]
    facets = _maximal(
        frozenset(np.where(cone.I[:, f])[0].tolist())
        for f in range(cone.I.shape[1])
        if cone.I[:, f].sum() >= n - 1
    )
    simps = _pull(list(range(m)), facets, n)
    if not simps:  # numerically lower-dimensional cone (e.g. a sliver of a near-degenerate split)
        return np.zeros((0, n, n))
    return cone.V[np.array(simps, dtype=int)]


class _PMap(np.ndarray):
    """Pre-activation map that carries its per-neuron norms (shared by all child cones)."""


def _preact_map(rnn, H, t, fl):
    """Pre-activation map at step t (d, n) given the hidden-state map H (only its first t
    columns are non-zero), with its row norms (used for sign tolerances) attached."""
    P = rnn.Wh @ H
    P[:, t] += rnn.Wi
    P = P.view(_PMap)
    P.sc = np.linalg.norm(P[:, : t + 1], axis=1) + 1e-300
    fl.rnn += 2 * rnn.d * rnn.d * t + 2 * rnn.d * (t + 1) + rnn.d
    return P


def _ray_preacts(V, P, t, fl):
    """Pre-activations of rays V under map P (only the first t + 1 columns are non-zero)."""
    fl.rnn += 2 * len(V) * (t + 1) * P.shape[0]
    return np.asarray(V[:, : t + 1] @ P[:, : t + 1].T)


def _signs(S, P, t, fl):
    """Per-neuron sign summary over the rays: (pos, neg) boolean (m, d) arrays."""
    m, d = S.shape
    fl.rnn += 4 * m * d + d  # two comparisons, two reductions, the conflict test
    return S > TOL * P.sc, S < -TOL * P.sc


def _split_children(cone, P, S, i, t, fl):
    """Split `cone` along neuron i's pre-activation hyperplane at step t. Children inherit the
    pre-activation map P and the cached pre-activations of the parent's rays; only the new rays
    are evaluated. Returns a list of (child, S_child)."""
    plus, minus, (Pi, Ni, Zi, W) = split(cone, P[i], fl, parts=True)
    SW = _ray_preacts(W, P, t, fl)
    return [
        (plus, np.concatenate([S[Pi], S[Zi], SW])),
        (minus, np.concatenate([S[Ni], S[Zi], SW])),
    ]


def linear_pieces(rnn, cone, fl):
    """Generator over (cone, L): an exact partition of `cone` into pieces on each of which the
    RNN is linear with logits L x. Cones are split along pre-activation hyperplanes, processing
    time steps in order (children inherit cached pre-activations, see _split_children)."""
    d, n = rnn.d, rnn.n
    stack = [(cone, 0, np.zeros((d, n)), None, None)]
    while stack:
        cone, t, H, P, S = stack.pop()
        split_done = False
        while t < n:
            if P is None:
                P = _preact_map(rnn, H, t, fl)
                S = _ray_preacts(cone.V, P, t, fl)
            pos, neg = _signs(S, P, t, fl)
            conflict = pos.any(0) & neg.any(0)
            if conflict.any():
                i = np.where(conflict)[0][0]
                for child, S_child in _split_children(cone, P, S, i, t, fl):
                    stack.append((child, t, H, P, S_child))
                split_done = True
                break
            H = pos.any(0)[:, None] * P
            fl.rnn += d * (t + 1)
            t += 1
            P = S = None
        if not split_done:
            fl.rnn += 2 * len(rnn.Wo) * d * n
            yield cone, rnn.Wo @ H


def cut_incorrect(cone, L, b, fl):
    """Pieces of `cone` (on which the logits are L x) where argmax_j L_j x != b, as a list of
    disjoint cones (first-index tie breaking for identical logits)."""
    lscale = np.abs(L).max() + 1e-300
    fl.split += 3 * L.size  # scale, differences and their magnitudes
    bad = []
    for j in range(len(L)):
        if j == b:
            continue
        dvec = L[b] - L[j]
        if np.abs(dvec).max() <= 1e-12 * lscale:
            if j < b:
                return bad + [cone]
            continue
        plus, minus = split(cone, dvec, fl)
        if minus is not None:
            bad.append(minus)
        if plus is None:
            return bad
        cone = plus
    return bad


def cut_correct(cone, L, b, fl):
    """The piece of `cone` where argmax_j L_j x == b (a single convex cone), or None."""
    lscale = np.abs(L).max() + 1e-300
    for j in range(len(L)):
        if j == b:
            continue
        dvec = L[b] - L[j]
        if np.abs(dvec).max() <= 1e-12 * lscale:
            if j < b:
                return None
            continue
        cone, _ = split(cone, dvec, fl)
        if cone is None:
            return None
    return cone


def tie_normals(R, task):
    """Normals e_i - e_j of the root's facets on which the target changes (ties between the
    target rank and an adjacent rank)."""
    n = R.shape[1]
    order = np.argsort(-R.sum(0))
    r = target_rank(task, n)
    out = []
    for a, b in ((r - 1, r), (r, r + 1)):
        if 0 <= a and b < n:
            c = np.zeros(n)
            c[order[a]], c[order[b]] = 1 / math.sqrt(2), -1 / math.sqrt(2)
            out.append(c)
    return out


def measure_simplices(S, n, fl, measure="slice", s=3, n_quad=20, max_angle=0.3, tiny=1e-6):
    """Gaussian measures of simplicial cones S (B, n, n).

    n <= 3: closed-form orthant probabilities (exact). n >= 4, measure="slice" (default):
      * narrow cones (rays within max_angle radians): slice cubature of degree 2s+1;
      * wide cones: Plackett's formula with n_quad nodes per panel, which is accurate for fat
        cones but loses absolute accuracy (~1e-9) on very thin ones through cancellation, so
        wide cones it measures below `tiny` are recomputed by slice cubature after bisection.
      With the defaults the result is converged to ~1e-11 on the zoo models (raising s, lowering
      max_angle or tiny changes nothing at that level).
    measure="orthant": Plackett throughout (an independent check)."""
    out = np.zeros(len(S))
    if len(S) == 0:
        return out
    use_slice = np.zeros(len(S), bool)
    if n >= 4 and measure == "slice":
        U = S / np.linalg.norm(S, axis=-1, keepdims=True)
        use_slice = (U @ np.swapaxes(U, -1, -2)).min((-1, -2)) >= math.cos(max_angle)
        fl.measure += len(S) * 2 * n**3
    for k in range(0, len(S), 8192):
        sl = slice(k, k + 8192)
        nr, chunk = use_slice[sl], S[sl]
        vals = np.zeros(len(chunk))
        if nr.any():
            v, pieces = slice_measure(chunk[nr], s=s, max_angle=max_angle, return_pieces=True)
            vals[nr] = v
            fl.measure += slice_flops(n, s) * pieces
        if (~nr).any():
            wide = chunk[~nr]
            v = cone_measure(wide, n_quad=n_quad)
            fl.measure += orthant_flops(n, n_quad) * len(wide)
            redo = (v < tiny) & (n >= 4) & (measure == "slice")
            if redo.any():
                v2, pieces = slice_measure(wide[redo], s=s, max_angle=max_angle, return_pieces=True)
                v[redo] = v2
                fl.measure += slice_flops(n, s) * pieces
            vals[~nr] = v
        out[sl] = vals
    return out


def exact_accuracy(rnn, task, measure="slice", s=3, n_quad=20):
    """Accuracy of an RNN classifier on N(0, I) inputs, exact up to quadrature / float error:
    the sum over the 2 n! root cones of root_accuracy. Returns (accuracy, Flops)."""
    fl = Flops()
    acc = 0.0
    for root in root_cones(rnn.n, task):
        a, f = _root_accuracy(rnn, *root, measure=measure, s=s, n_quad=n_quad)
        acc += a
        fl.add(f)
    return acc, fl


def root_accuracy(rnn, task, k, measure="slice", s=3, n_quad=20):
    """Contribution of the k-th root cone (in root_cones order) to the accuracy, i.e. the
    Gaussian measure of the part of that root where the model is correct. Roots are
    independent, so they can be computed in parallel (exact_parallel.py). Returns
    (contribution, Flops)."""
    root = next(itertools.islice(root_cones(rnn.n, task), k, None))
    return _root_accuracy(rnn, *root, measure=measure, s=s, n_quad=n_quad)


def _root_accuracy(rnn, R, b, root_meas, measure="slice", s=3, n_quad=20, batch=1 << 16):
    """We measure whichever side of the root is likely smaller (decided by the prediction at
    its central ray): either the root's measure minus the incorrect pieces, or the correct
    pieces. Regions on which the model is entirely correct (or entirely wrong) therefore cost no
    measure computation. Simplices are measured in batches of about `batch` as they are
    produced, so memory stays bounded."""
    fl = Flops()
    n = rnn.n
    fl.rnn += rnn.forward_flops()
    count_good = int(rnn.forward(R.sum(0)).argmax()) != b
    sign = 1.0 if count_good else -1.0
    acc = 0.0 if count_good else root_meas
    pending, size = [], 0

    def flush():
        S = np.concatenate(pending) if pending else np.zeros((0, n, n))
        fl.counts["simplices"] += len(S)
        pending.clear()
        return sign * measure_simplices(S, n, fl, measure, s, n_quad).sum()

    for cone, L in linear_pieces(rnn, Cone.simplicial(R), fl):
        fl.counts["leaves"] += 1
        if count_good:
            pieces = [p for p in [cut_correct(cone, L, b, fl)] if p is not None]
        else:
            pieces = cut_incorrect(cone, L, b, fl)
        for piece in pieces:
            T = triangulate(piece)
            pending.append(T)
            size += len(T)
            fl.split += 5 * len(piece.V) * n  # triangulation bookkeeping (traced)
        if size >= batch:
            acc += flush()
            size = 0
    acc += flush()
    return acc, fl


def _resolve_rest(rnn, cone, t, H, P, S, point, fl):
    """Finish the time steps on `cone` without splitting, starting from step t whose
    pre-activation map P and ray pre-activations S are already known: every conflicting neuron
    takes its sign at `point`. Returns the (approximate) logits map."""
    d, n = rnn.d, rnn.n
    while True:
        pos, neg = _signs(S, P, t, fl)
        conflict = pos.any(0) & neg.any(0)
        at_point = point[: t + 1] @ P[:, : t + 1].T > 0
        fl.rnn += 2 * (t + 1) * d + d * (t + 1)
        H = np.where(conflict, at_point, pos.any(0))[:, None] * P
        t += 1
        if t == n:
            break
        P = _preact_map(rnn, H, t, fl)
        S = _ray_preacts(cone.V, P, t, fl)
    fl.rnn += 2 * len(rnn.Wo) * d * n
    return rnn.Wo @ H


def _tie_point(cone, ties):
    """A point just inside the cone next to a face it shares with a target tie facet (where its
    errors would be), or its central ray if it has no such face."""
    n = cone.V.shape[1]
    center = cone.V.sum(0)
    for c in ties:
        on = np.abs(cone.V @ c) <= 1e-9
        if on.sum() >= n - 1:
            return cone.V[on].mean(0) + 1e-3 * center / len(cone.V)
    return center


def _root_setup_flops(n):
    """Cofactor facet normals of a simplicial root cone plus its tie normals (an argsort)."""
    return 4 * n**3 + 10 * n


def _heap_flops(size):
    return 2 * max(1, math.ceil(math.log2(size + 1)))


def _finish(rnn, cone, t, H, P, S, b, ties, fl, rule="guard", tau=0.1, big=0.01, measure="slice",
            s=3, n_quad=20):
    """Incorrect measure of a cone popped from the refinement queue that will not be split any
    further: advance it exactly while its rays agree; if it reaches the end it is a single linear
    region and its incorrect part is exact. Otherwise it is linearised at the first conflict:

      rule="tie":   conflicting neurons take their sign at the tie point (see _tie_point);
      rule="guard": as "tie", but the cone is also linearised at its central ray, and if the two
                    estimates of its incorrect measure differ by more than tau times the cone's
                    measure the smaller one is used. A linearisation claiming a large part of a
                    cone is wrong when another does not is unreliable, and the smaller estimate
                    is the one closer to the prior that trained models are almost always right.
      rule="cguard": as "guard", but the second linearisation is only computed when the tie
                    estimate claims more than tau of the cone is wrong (so plausible cones cost
                    no more than with "tie").
      rule="lguard": as "cguard", but only for large cones (measure >= big), where a failed
                    linearisation is costly; small cones keep the tie estimate.
    """
    d, n = rnn.d, rnn.n

    def incorrect(L):
        pieces = cut_incorrect(cone, L, b, fl)
        if not pieces:
            return 0.0
        simps = np.concatenate([triangulate(p) for p in pieces])
        fl.split += 5 * sum(len(p.V) for p in pieces) * n
        return float(measure_simplices(simps, n, fl, measure, s, n_quad).sum())

    while t < n:
        if P is None:
            P = _preact_map(rnn, H, t, fl)
            S = _ray_preacts(cone.V, P, t, fl)
        pos, neg = _signs(S, P, t, fl)
        if (pos.any(0) & neg.any(0)).any():
            fl.rnn += 2 * len(cone.V) * n * (len(ties) + 1)
            e_tie = incorrect(_resolve_rest(rnn, cone, t, H, P, S, _tie_point(cone, ties), fl))
            if rule == "tie" or e_tie == 0.0:
                return e_tie
            simps = triangulate(cone)
            fl.split += 5 * len(cone.V) * n
            mu = float(measure_simplices(simps, n, fl, measure, s, n_quad).sum())
            if rule in ("cguard", "lguard") and e_tie <= tau * mu:
                return e_tie  # plausible: only claims a small part of the cone is wrong
            if rule == "lguard" and mu < big:
                return e_tie  # a small cone: a wrong linearisation can only cost its measure
            e_ctr = incorrect(_resolve_rest(rnn, cone, t, H, P, S, cone.V.sum(0), fl))
            return min(e_tie, e_ctr) if abs(e_tie - e_ctr) > tau * mu else e_tie
        H = pos.any(0)[:, None] * P
        fl.rnn += d * (t + 1)
        t += 1
        P = S = None
    return _region_incorrect(rnn, cone, H, b, fl, measure, s, n_quad)


def _region_incorrect(rnn, cone, H, b, fl, measure="slice", s=3, n_quad=20, **_):
    """Exact incorrect measure of a cone on which the RNN is linear with hidden map H."""
    n = rnn.n
    fl.rnn += 2 * len(rnn.Wo) * rnn.d * n
    pieces = cut_incorrect(cone, rnn.Wo @ H, b, fl)
    if not pieces:
        return 0.0
    simps = np.concatenate([triangulate(p) for p in pieces])
    fl.split += 5 * sum(len(p.V) for p in pieces) * n
    return float(measure_simplices(simps, n, fl, measure, s, n_quad).sum())


def anytime_accuracy(rnn, task, max_splits, max_roots=None, rule="guard", **kw):
    """Deterministic accuracy estimate with a compute knob (reference implementation of one rung
    of anytime_ladder). Cones are refined exactly in order of decreasing angular width (a global
    priority queue over all root cones) until max_splits splits have been made; every cone left
    is finished by _finish (exact if it is a single linear region, else linearised). Exact once
    max_splits is large enough. Measures the incorrect side (intended for accuracy > 1/2).

    max_roots < 2 n! gives the cheapest rung: only the first max_roots root cones are analysed
    (with no splits) and the rest are left at the prior "correct", in the spirit of treating
    unexamined cases at their prior rather than extrapolating from examined ones."""
    d, n = rnn.d, rnn.n
    fl = Flops()
    heap = []
    counter = itertools.count()

    def width(cone):
        fl.split += 2 * len(cone.V) ** 2 * n
        return -float((cone.V @ cone.V.T).min())  # larger = wider (min cosine between rays)

    roots = list(root_cones(n, task))
    if max_roots is not None:
        roots = roots[:max_roots]
    for R, b, _ in roots:
        cone = Cone.simplicial(R)
        fl.split += _root_setup_flops(n) + _heap_flops(len(heap))
        root = (cone, 0, np.zeros((d, n)), None, None, b, tie_normals(R, task))
        heapq.heappush(heap, (-width(cone), next(counter)) + root)
    splits = 0
    err = 0.0
    while heap:
        fl.split += _heap_flops(len(heap))
        _, _, cone, t, H, P, S, b, ties = heapq.heappop(heap)
        if splits < max_splits:
            item = _advance_to_conflict(rnn, cone, t, H, P, S, fl)
            if item[0] == "done":  # a single linear region: exact
                err += _region_incorrect(rnn, cone, item[1], b, fl, **kw)
                continue
            _, t, H, P, S, i = item
            for child, S_child in _split_children(cone, P, S, i, t, fl):
                fl.split += _heap_flops(len(heap))
                key = (-width(child), next(counter))
                heapq.heappush(heap, key + (child, t, H, P, S_child, b, ties))
            splits += 1
        else:
            err += _finish(rnn, cone, t, H, P, S, b, ties, fl, rule, **kw)
    fl.counts["splits"] = splits
    return 1.0 - err, fl


def _advance_to_conflict(rnn, cone, t, H, P, S, fl):
    """Advance a cone exactly while its rays agree on every activation. Returns ("done", H) if
    it reaches the end (a single linear region with hidden map H), else
    ("conflict", t, H, P, S, neuron) at the first conflict."""
    d, n = rnn.d, rnn.n
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


def anytime_ladder(rnn, task, split_grid, root_grid=(), rule="guard", **kw):
    """All rungs of the estimator ladder in one refinement pass. Returns a dict
    name -> (accuracy estimate, flops) where "roots<k>" analyses the first k root cones with no
    splits (rest at the prior) and "any<k>" allows k splits. Estimates equal those of standalone
    anytime_accuracy runs; flops equal them exactly for "any<k>" rungs and exceed them by a few
    heap comparisons (<= ~16 flops) for "roots<k>" rungs."""
    d, n = rnn.d, rnn.n
    out = {}
    roots = list(root_cones(n, task))
    # roots rungs: per-root cost and contribution with no splits, accumulated in root order
    cum_err, cum_fl = 0.0, 0.0
    per_root = []
    for k, (R, b, _) in enumerate(roots):
        f = Flops()
        cone = Cone.simplicial(R)
        f.split += _root_setup_flops(n) + _heap_flops(k) + 2 * len(cone.V) ** 2 * n
        f.split += _heap_flops(len(roots) - k)  # its pop (heap order does not change the sum)
        e = _finish(rnn, cone, 0, np.zeros((d, n)), None, None, b, tie_normals(R, task), f,
                    rule, **kw)
        per_root.append((e, f.total))
    for k in sorted(set(root_grid)):
        if k < len(roots):
            out[f"roots{k}"] = (1.0 - sum(e for e, _ in per_root[:k]),
                                sum(c for _, c in per_root[:k]))
    # split rungs
    grid = sorted(set(split_grid))
    fl = Flops()
    heap = []
    counter = itertools.count()

    def width(cone):
        fl.split += 2 * len(cone.V) ** 2 * n
        return -float((cone.V @ cone.V.T).min())

    for R, b, _ in roots:
        cone = Cone.simplicial(R)
        fl.split += _root_setup_flops(n) + _heap_flops(len(heap))
        heapq.heappush(heap, (-width(cone), next(counter), cone, 0, np.zeros((d, n)), None, None,
                              b, tie_normals(R, task)))
    splits, err = 0, 0.0

    def snapshot():
        f = Flops()
        e = err
        size = len(heap)
        for item in heap:
            f.split += _heap_flops(size)
            size -= 1
            _, _, cone, t, H, P, S, b, ties = item
            e += _finish(rnn, cone, t, H, P, S, b, ties, f, rule, **kw)
        return 1.0 - e, fl.total + f.total

    gi = 0
    while gi < len(grid) and grid[gi] <= splits:
        out[f"any{grid[gi]}"] = snapshot()
        gi += 1
    while heap and gi < len(grid):
        fl.split += _heap_flops(len(heap))
        _, _, cone, t, H, P, S, b, ties = heapq.heappop(heap)
        item = _advance_to_conflict(rnn, cone, t, H, P, S, fl)
        if item[0] == "done":
            err += _region_incorrect(rnn, cone, item[1], b, fl, **kw)
            continue
        _, t2, H2, P2, S2, i = item
        for child, S_child in _split_children(cone, P2, S2, i, t2, fl):
            fl.split += _heap_flops(len(heap))
            heapq.heappush(heap, (-width(child), next(counter), child, t2, H2, P2, S_child, b, ties))
        splits += 1
        while gi < len(grid) and grid[gi] <= splits:
            out[f"any{grid[gi]}"] = snapshot()
            gi += 1
    if gi < len(grid):  # refinement finished early: remaining rungs are exact
        while heap:
            fl.split += _heap_flops(len(heap))
            _, _, cone, t, H, P, S, b, ties = heapq.heappop(heap)
            err += _finish(rnn, cone, t, H, P, S, b, ties, fl, rule, **kw)
        final = (1.0 - err, fl.total)
        for k in grid[gi:]:
            out[f"any{k}"] = final
    return out
