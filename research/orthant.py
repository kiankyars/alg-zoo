"""Gaussian measure of simplicial cones (orthant probabilities), dimensions 1-5.

A simplicial cone with extreme rays r_1..r_n (rows of R) is {x = R^T lam : lam >= 0} =
{x : a_i . x >= 0 for all i}, where a_i are the unit inward facet normals (rows of the cofactor
matrix of R, normalised). For x ~ N(0, I_n), z_i = a_i . x is N(0, C) with C_ij = a_i . a_j, so
the Gaussian measure of the cone is the orthant probability P(z >= 0).

n <= 3 use the classical closed forms. n = 4, 5 use Plackett's identity
    dP/dC_ij = phi_2(0, 0; C_ij) * P(z_rest >= 0 | z_i = z_j = 0)
integrated along C(t) = I + t (C - I); the conditional orthant probability has dimension
n - 2 <= 3 and so has a closed form. All quantities that become ill-conditioned for thin cones
(1 -+ rho and the conditional covariance) are computed directly from the facet normals, and the
substitution t = 1 - s^2 plus Gauss-Legendre panels graded towards s = 0 (only as deep as the
cone's thinness requires) resolves the near-singularity at t = 1.
"""

import math

import numpy as np

_GL = {}


def _rule(q):
    if q not in _GL:
        x, w = np.polynomial.legendre.leggauss(q)
        _GL[q] = (0.5 * (x + 1.0), 0.5 * w)
    return _GL[q]


PANEL_RATIO = 3.0


def _panels(levels, q):
    """Nodes/weights on [0, 1]: panels [0, r^-L], [r^-L, r^-(L-1)], ..., [r^-1, 1]."""
    x, w = _rule(q)
    edges = np.concatenate([[0.0], PANEL_RATIO ** -np.arange(levels, -1, -1.0)])
    a, b = edges[:-1, None], edges[1:, None]
    return (a + (b - a) * x).ravel(), ((b - a) * w).ravel()


def _asin(x):
    return np.arcsin(np.clip(x, -1.0, 1.0))


def _closed_form(C):
    n = C.shape[-1]
    if n == 1:
        return np.full(C.shape[:-2], 0.5)
    if n == 2:
        return 0.25 + _asin(C[..., 0, 1]) / (2 * math.pi)
    if n == 3:
        return 0.125 + (_asin(C[..., 0, 1]) + _asin(C[..., 0, 2]) + _asin(C[..., 1, 2])) / (
            4 * math.pi
        )
    raise NotImplementedError(n)


def orthant_from_normals(A, n_quad=12):
    """P(A x >= 0) for x ~ N(0, I), A of shape (B, n, n) with unit rows (facet normals)."""
    B, n, _ = A.shape
    C = A @ np.swapaxes(A, -1, -2)
    if n <= 3:
        return _closed_form(C)
    if n > 5:
        raise NotImplementedError(n)
    # thinness: 1 - |C_ij| computed accurately as min(|a_i - a_j|^2, |a_i + a_j|^2) / 2
    diff = A[:, :, None, :] - A[:, None, :, :]
    summ = A[:, :, None, :] + A[:, None, :, :]
    dm = 0.5 * (diff**2).sum(-1)  # = 1 - C_ij
    dp = 0.5 * (summ**2).sum(-1)  # = 1 + C_ij
    iu = np.triu_indices(n, 1)
    delta = np.minimum(dm, dp)[:, iu[0], iu[1]].min(1)
    # grade panels down to ~ sqrt(delta) / 10, the scale of the near-singularity in s
    levels = np.clip(np.ceil(np.log(10.0 / np.sqrt(np.maximum(delta, 1e-30))) / np.log(PANEL_RATIO)), 2, 40)
    out = np.empty(B)
    for L in np.unique(levels):
        sel = levels == L
        out[sel] = _plackett(A[sel], C[sel], dm[sel], dp[sel], *_panels(int(L), n_quad))
    return out


def _plackett(A, C, dm, dp, s, ws):
    B, n, _ = A.shape
    t = 1.0 - s**2
    dt = 2.0 * s * ws
    total = np.zeros(B)
    for i in range(n):
        for j in range(i + 1, n):
            rest = [k for k in range(n) if k not in (i, j)]
            cij = C[:, i, j]
            # 1 - rho and 1 + rho for rho = t * cij, without cancellation
            pos = (cij >= 0)[:, None]
            one_minus = np.where(pos, dm[:, i, j][:, None] * t + s**2, 1.0 - t * cij[:, None])
            one_plus = np.where(~pos, dp[:, i, j][:, None] * t + s**2, 1.0 + t * cij[:, None])
            # (careful: 1 - t c = (1 - t) + t (1 - c) and 1 - t = s^2)
            dens = 1.0 / (2 * math.pi * np.sqrt(one_minus * one_plus))
            # conditional covariance of rest given z_i = z_j = 0 under C(t) = I + t (C - I):
            # S = I_rest + t (C_rest - I) - u u^T - v v^T with
            # u = (b_i + b_j) / sqrt(2 (1 + rho)), v = (b_i - b_j) / sqrt(2 (1 - rho)),
            # b_i = t C[rest, i] = t A_rest a_i.
            Ar = A[:, rest]  # (B, m, n)
            bp = Ar @ (A[:, i] + A[:, j])[..., None]  # (B, m, 1)
            bm = Ar @ (A[:, i] - A[:, j])[..., None]
            u = t[None, :, None] * bp[:, None, :, 0] / np.sqrt(2 * one_plus)[..., None]
            v = t[None, :, None] * bm[:, None, :, 0] / np.sqrt(2 * one_minus)[..., None]
            Cr = C[:, rest][:, :, rest]
            m = len(rest)
            S = (np.eye(m) + t[None, :, None, None] * (Cr - np.eye(m))[:, None]) - (
                u[..., :, None] * u[..., None, :] + v[..., :, None] * v[..., None, :]
            )
            dg = np.sqrt(np.clip(np.diagonal(S, axis1=-2, axis2=-1), 1e-300, None))
            pc = _closed_form(S / (dg[..., :, None] * dg[..., None, :]))
            total += cij * (dens * pc * dt[None, :]).sum(1)
    return np.clip(2.0 ** (-n) + total, 0.0, 0.5)


def orthant_flops(n, n_quad=12, levels=4):
    """Rough flop count of cone_measure for one simplicial cone in dimension n."""
    if n <= 3:  # 2x2 / 1x1 cofactors, normalisation, normal Gram matrix, asin terms (20 each)
        pairs = n * (n - 1) // 2
        return 2 * n * n * (n - 1) + n * (2 * n + 10) + pairs * (2 * n - 1) + 20 * pairs + 5
    base = n * n * (n - 1) ** 3 + 4 * n**3  # cofactors + normal Gram matrix
    nodes = (levels + 1) * n_quad
    m = n - 2
    per_node = 6 * m * n + 4 * m * m + 30 * (m * (m - 1) // 2) + 20  # u, v, S, asin terms, dens
    return base + 6 * n**3 + (n * (n - 1) // 2) * (nodes * per_node + 4 * m * n)


def normals(R):
    """Unit inward facet normals (rows) of simplicial cones with rays R (B, n, n), via
    cofactors, which stay well conditioned as the cone becomes thin. Returns (A, ok)."""
    B, n, _ = R.shape
    R = R / np.linalg.norm(R, axis=-1, keepdims=True)
    if n == 1:
        return np.sign(R), np.ones(B, bool)
    cof = np.empty_like(R)
    idx = np.arange(n)
    for i in range(n):
        for j in range(n):
            cof[:, i, j] = (-1) ** (i + j) * np.linalg.det(R[:, idx != i][:, :, idx != j])
    norms = np.linalg.norm(cof, axis=-1)
    ok = (norms > 1e-300).all(-1) & (np.abs(np.linalg.det(R)) > 1e-15)
    A = cof / np.where(norms > 0, norms, 1.0)[..., None]
    A *= np.sign((A * R).sum(-1, keepdims=True) + 1e-300)  # orient inwards
    return A, ok


def cone_measure(R, n_quad=12):
    """Gaussian measure of simplicial cones with rays R of shape (B, n, n) (rows are rays)."""
    R = np.asarray(R, dtype=np.float64)
    B, n, _ = R.shape
    if n == 1:
        return np.full(B, 0.5)
    A, ok = normals(R)
    out = np.zeros(B)
    if ok.any():
        out[ok] = np.clip(orthant_from_normals(A[ok], n_quad=n_quad), 0.0, None)
    return out
