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


def _narrow_pieces(R, max_angle):
    """Bisects simplicial cones R (B, n, n) along their longest edge until all their rays are
    within max_angle radians. Returns [(rays (b, n, n), index of the original cone (b,))]."""
    R = np.asarray(R, dtype=np.float64)
    R = R / np.linalg.norm(R, axis=-1, keepdims=True)
    B, n, _ = R.shape
    work = [(R, np.arange(B))]
    leaves = []
    while work:
        Rs, idx = work.pop()
        G = Rs @ np.swapaxes(Rs, -1, -2)
        flat = np.arccos(np.clip(G, -1, 1)).reshape(len(Rs), -1)
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
    return leaves


def slice_linear_moment(R, w, s=3, max_angle=0.3, return_pieces=False):
    """E[(w . x) 1_K(x)] for x ~ N(0, I_n) and simplicial cones K with rays R (B, n, n). In polar
    coordinates it is C_n * int_{K on the sphere} (w . u) du with
    C_n = (2 pi)^(-n/2) 2^((n-1)/2) Gamma((n+1)/2), and on the slice {y : u0 . y = 1}
    du = |y|^-n dA, so it is C_n int_slice (w . y) |y|^-(n+1) dA: a smooth integrand on a flat
    simplex, integrated by GM cubature of degree 2s+1 after bisecting wide cones. Unlike the
    integration-by-parts formula (a signed sum of facet measures), it stays accurate on flat
    cones, whose facet measures nearly cancel."""
    B, n, _ = np.shape(R)
    out = np.zeros(B)
    leaves = _narrow_pieces(R, max_angle)
    P, W = gm_rule(n - 1, s)
    C = (2 * math.pi) ** (-n / 2) * 2 ** ((n - 1) / 2) * math.gamma((n + 1) / 2)
    for Rs, idx in leaves:
        u = Rs.sum(1)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        Y = Rs / (Rs @ u[:, :, None])  # slice vertices (b, n, n)
        pts = np.einsum("pj,bjk->bpk", P, Y)  # (b, P, n)
        f = (pts @ w) * (pts**2).sum(-1) ** (-(n + 1) / 2)
        vals = C * np.abs(np.linalg.det(Y)) / math.factorial(n - 1) * (f @ W)
        np.add.at(out, idx, vals)
    if return_pieces:
        return out, sum(len(i) for _, i in leaves)
    return out


def _theta(m, a):
    """int_0^{atan a} cos^(m-1) t dt (a >= 0), in closed form for m <= 4."""
    a = np.maximum(a, 0.0)
    if m == 1:
        return np.arctan(a)
    s = a / np.sqrt(1 + a * a)  # sin(atan a)
    if m == 2:
        return s
    if m == 3:
        return np.arctan(a) / 2 + a / (2 * (1 + a * a))
    if m == 4:
        return s - s**3 / 3
    raise NotImplementedError(m)


def slice_sliver_measure(R, w, s=3, max_angle=0.3, return_pieces=False):
    """Gaussian measure (in R^(m+1)) of the slivers {zeta + tau nu : zeta in K, 0 <= tau <= w . zeta}
    over simplicial cones K with rays R (B, m, m) in an m-dim subspace, nu a unit normal to it and
    w a linear functional that is nonnegative on K. In polar coordinates zeta = rho u, the sliver
    over the ray u is the 2-d wedge {(rho, tau) : 0 <= tau <= rho (w . u)}, whose Gaussian measure
    with the radial weight rho^(m-1) is C_(m+1) Theta_m(w . u), Theta_m(a) = int_0^{atan a}
    cos^(m-1) t dt and C_(m+1) = (2 pi)^(-(m+1)/2) 2^((m-1)/2) Gamma((m+1)/2). So the measure is
    C int_slice Theta_m(w . y / |y|) |y|^-m dA, integrated by GM cubature as in
    slice_linear_moment. For thin slivers Theta_m(a) ~ a and it reduces to
    phi(0) E[(w . zeta) 1_K]; unlike that, it keeps the decay of the Gaussian density across
    the sliver, so it is exact for wide slivers too."""
    R = np.asarray(R, dtype=np.float64)
    B, m, _ = R.shape
    C = (2 * math.pi) ** (-(m + 1) / 2) * 2 ** ((m - 1) / 2) * math.gamma((m + 1) / 2)
    if m == 1:  # K is a ray u: the sliver is one 2-d wedge
        u = R[:, 0] / np.linalg.norm(R[:, 0], axis=-1, keepdims=True)
        out = C * _theta(1, u @ w)
        return (out, B) if return_pieces else out
    out = np.zeros(B)
    leaves = _narrow_pieces(R, max_angle)
    P, W = gm_rule(m - 1, s)
    for Rs, idx in leaves:
        u = Rs.sum(1)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        Y = Rs / (Rs @ u[:, :, None])  # slice vertices (b, m, m)
        pts = np.einsum("pj,bjk->bpk", P, Y)
        r = np.sqrt((pts**2).sum(-1))
        f = _theta(m, (pts @ w) / r) * r ** (-m)
        vals = C * np.abs(np.linalg.det(Y)) / math.factorial(m - 1) * (f @ W)
        np.add.at(out, idx, vals)
    if return_pieces:
        return out, sum(len(i) for _, i in leaves)
    return out


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
