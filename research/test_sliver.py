"""Check the tie-sliver laws: the exact (cone-geometry) laws of sliver.py against independent Monte
Carlo estimates of the same formulas (sliver_mc.py), and against the exact error.

* the sliver measure (slice.slice_sliver_measure) vs direct sampling of random slivers;
* first-order and order-truncated laws vs sliver_mc.law (finite differences on sampled tie points);
* the exact sliver law (bisector cap, exact widths) vs sliver_mc.widths (true widths found by a
  grid and bisection along sampled normal lines, with numpy's argmax, i.e. ties to the lowest index).

Usage: python test_sliver.py M:8,3 M:16,4 ...   (default: a few fast models)"""
import sys
import time

import numpy as np
from scipy.special import ndtr

from engine import RNN, exact_accuracy
from evaluate import load
from slice import slice_sliver_measure
from sliver import sliver_error
from sliver_mc import law, widths

rng = np.random.default_rng(0)
for m in (1, 2, 3, 4):  # slivers {zeta + tau nu : zeta in K, 0 <= tau <= w . zeta} in R^(m+1)
    R = rng.standard_normal((1, m, m)) + 1.5  # a random simplicial cone (rows are rays)
    w = np.linalg.solve(R[0], rng.uniform(0.1, 3.0, m))  # w . ray in [0.1, 3]: thin to wide slivers
    exact = slice_sliver_measure(R, w)[0]
    # Monte Carlo over zeta of P(0 <= tau <= w . zeta) = Phi(w . zeta) - 1/2 on K (no cubature, no polar form)
    N, s1, s2 = 1 << 24, 0.0, 0.0
    for _ in range(16):
        Z = rng.standard_normal((N // 16, m))
        inside = (np.linalg.solve(R[0].T, Z.T) >= 0).all(0)  # zeta = coef @ rays with coef >= 0
        v = np.where(inside, ndtr(Z @ w) - 0.5, 0.0)
        s1, s2 = s1 + v.sum(), s2 + (v * v).sum()
    mc = s1 / N
    se = np.sqrt((s2 / N - mc * mc) / N)
    print(f"sliver measure, m = {m}: {exact:.6e} | Monte Carlo {mc:.6e} ± {se:.1e} (z = {(exact - mc) / se:+.2f})",
          flush=True)

# a toy RNN where h_1 = 0 on a cone, so that every step-2 neuron flips, and every logit gap vanishes,
# on x_2 = 0 at once (the walk must apply all simultaneous flips before judging the gaps)
toy = RNN(Wi=np.array([1.0, -1.0]), Wh=np.array([[-20.0, 0.0], [0.0, 0.0]]),
          Wo=np.array([[0.0, 0.0], [1.0, -0.3], [0.2, 1.0]]), n=3)
e = sliver_error(toy, "2nd_argmax", cap="bisector", width="exact")[0]
mc, se = widths(toy, "2nd_argmax", 1 << 16, rng)["bisector"]
print(f"toy RNN exact sliver: law {e:.6e} | Monte Carlo {mc:.6e} ± {se:.1e} (z = {(e - mc) / se:+.2f})", flush=True)

for spec in sys.argv[1:] or ["M:2,2", "M:3,3", "M:4,3", "M:8,3", "M:32,3", "M:8,4", "M:16,4"]:
    rnn, (task, _) = load(spec)
    fwd = rnn.forward_flops()
    t0 = time.time()
    acc, fle = exact_accuracy(rnn, task)
    ex = 1 - acc
    (mc1, se1), (mc2, se2) = law(rnn, task, 1 << 20, np.random.default_rng(0))
    mcw = widths(rnn, task, 1 << 15, np.random.default_rng(1)) if rnn.n >= 3 else None
    for name, cap, width, ref in (("first-order", "none", "first", (mc1, se1)),
                                  ("truncated", "chamber", "first", (mc2, se2)),
                                  ("exact sliver", "bisector", "exact", None if mcw is None else mcw["bisector"])):
        e, fls = sliver_error(rnn, task, cap=cap, width=width)
        line = f"{spec} {name:12s}: law {e:.6e} (rel to exact error {(e - ex) / ex:+.2%}, {fls.total / fwd:.3g} fwd)"
        if ref is not None:
            mc, se = ref
            line += f" | Monte Carlo {mc:.6e} ± {se:.1e} (z = {(e - mc) / se if se > 0 else 0.0:+.2f})"
        print(line, flush=True)
    print(f"{spec} exact error {ex:.6e} ({fle.total / fwd:.3g} fwd), {time.time() - t0:.0f}s", flush=True)
