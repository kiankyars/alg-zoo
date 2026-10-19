"""Gaussian measure of simplicial cones by cubature on a flat slice.

For a cone K = cone(r_1..r_n) and a unit interior direction u, intersect K with the affine
hyperplane {y : u.y = 1}. The solid angle subtended by a surface element dA at y is
dA / |y|^n, so
    gamma_n(K) = (1 / |S^{n-1}|) * int_{slice} |y|^{-n} dA(y),
an integral of a smooth (analytic) function over a flat (n-1)-simplex with vertices
y_j = r_j / (u.r_j). Grundmann-Moller cubature of degree 2s+1 integrates it to relative error
O(diam^{2s+2}), so small cones (the pieces we need) are measured very accurately at a cost
of a few dozen flops per cubature point. Large cones are recursively bisected first.
"""

import itertools
import math

import numpy as np

_GM = {}


def gm_rule(m, s):
    """Grundmann-Moller rule of degree 2s+1 on the m-simplex: barycentric points (P, m+1)
    and weights (P,) normalised to sum to 1."""
    key = (m, s)
    if key not in _GM:
        d = 2 * s + 1
        pts, wts = [], []
        for i in range(s + 1):
            w = (-1) ** i * 2.0 ** (-2 * s) * (d + m - 2 * i) ** d / (
                math.factorial(i) * math.factorial(d + m - i)
            )
            for beta in _compositions(s - i, m + 1):
                pts.append([(2 * b + 1) / (d + m - 2 * i) for b in beta])
                wts.append(w)
        wts = np.array(wts) * math.factorial(m)
        _GM[key] = (np.array(pts), wts / wts.sum())
    return _GM[key]


def _compositions(total, parts):
    for c in itertools.combinations(range(total + parts - 1), parts - 1):
        prev = -1
        out = []
        for x in c:
            out.append(x - prev - 1)
            prev = x
        out.append(total + parts - 1 - prev - 1)
        yield out


def sphere_area(n):
    return 2 * math.pi ** (n / 2) / math.gamma(n / 2)


def slice_measure(R, s=2, max_angle=0.5, return_pieces=False):
    """Gaussian measure of simplicial cones with rays R (B, n, n), by GM cubature of degree
    2s+1 on the slice. Cones whose rays span an angle > max_angle (radians) are bisected
    along their longest edge first."""
    R = np.asarray(R, dtype=np.float64)
    R = R / np.linalg.norm(R, axis=-1, keepdims=True)
    B, n, _ = R.shape
    out = np.zeros(B)
    # recursive bisection of wide cones (vectorised by rounds)
    work = [(R, np.arange(B))]
    leaves = []
    while work:
        Rs, idx = work.pop()
        G = Rs @ np.swapaxes(Rs, -1, -2)
        ang = np.arccos(np.clip(G, -1, 1))
        flat = ang.reshape(len(Rs), -1)
        wide = flat.max(1) > max_angle
        if (~wide).any():
            leaves.append((Rs[~wide], idx[~wide]))
        if wide.any():
            Rw, iw = Rs[wide], idx[wide]
            e = flat[wide].argmax(1)
            a, b = e // n, e % n
            mid = Rw[np.arange(len(Rw)), a] + Rw[np.arange(len(Rw)), b]
            mid /= np.linalg.norm(mid, axis=1, keepdims=True)
            R1, R2 = Rw.copy(), Rw.copy()
            R1[np.arange(len(Rw)), a] = mid
            R2[np.arange(len(Rw)), b] = mid
            work.append((np.concatenate([R1, R2]), np.concatenate([iw, iw])))
    P, W = gm_rule(n - 1, s)
    for Rs, idx in leaves:
        u = Rs.sum(1)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        Y = Rs / (Rs @ u[:, :, None])  # slice vertices (b, n, n)
        pts = np.einsum("pj,bjk->bpk", P, Y)  # (b, P, n)
        f = (pts**2).sum(-1) ** (-n / 2)
        vals = np.abs(np.linalg.det(Y)) / math.factorial(n - 1) * (f @ W) / sphere_area(n)
        np.add.at(out, idx, vals)
    if return_pieces:
        return out, sum(len(i) for _, i in leaves)
    return out


def slice_flops(n, s=2):
    """Rough flop count per cubature piece (after any bisection), including the angle check."""
    P = len(gm_rule(n - 1, s)[0])
    return 3 * n**3 + 4 * n * n + P * (2 * n * n + 3 * n + 10) + 2 * P
