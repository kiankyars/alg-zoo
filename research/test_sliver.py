"""Check the tie-sliver laws: the exact (cone-geometry) first-order and order-truncated laws of
sliver.py against an independent Monte Carlo estimate of the same laws (sliver_mc.py), and
against the exact error.
Usage: python test_sliver.py M:8,3 M:16,4 ...   (default: a few fast models)"""
import sys
import time

import numpy as np

from engine import exact_accuracy
from evaluate import load
from sliver import sliver_error
from sliver_mc import law

for spec in sys.argv[1:] or ["M:2,2", "M:8,3", "M:32,3", "M:8,4", "M:16,4"]:
    rnn, (task, _) = load(spec)
    fwd = rnn.forward_flops()
    t0 = time.time()
    acc, fle = exact_accuracy(rnn, task)
    ex = 1 - acc
    (mc1, se1), (mc2, se2) = law(rnn, task, 1 << 20, np.random.default_rng(0))
    for name, cap, mc, se in (("first-order", False, mc1, se1), ("truncated", True, mc2, se2)):
        e, fls = sliver_error(rnn, task, order_cap=cap)
        z = (e - mc) / se if se > 0 else 0.0
        print(f"{spec} {name:11s}: law {e:.6e} (rel to exact error {(e - ex) / ex:+.2%}, "
              f"{fls.total / fwd:.3g} fwd) | Monte Carlo law {mc:.6e} ± {se:.1e} (z = {z:+.2f})", flush=True)
    print(f"{spec} exact error {ex:.6e} ({fle.total / fwd:.3g} fwd), {time.time() - t0:.0f}s", flush=True)
