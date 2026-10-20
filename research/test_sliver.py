"""Check the tie-sliver laws: the exact (cone-geometry) laws of sliver.py against independent Monte
Carlo estimates of the same formulas (sliver_mc.py), and against the exact error.

* first-order and order-truncated laws vs sliver_mc.law (finite differences on sampled tie points);
* the exact sliver law (bisector cap, exact widths) vs sliver_mc.widths (true widths found by a
  grid and bisection along sampled normal lines, with numpy's argmax, i.e. ties to the lowest index).

Usage: python test_sliver.py M:8,3 M:16,4 ...   (default: a few fast models)"""
import sys
import time

import numpy as np

from engine import exact_accuracy
from evaluate import load
from sliver import sliver_error
from sliver_mc import law, widths

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
