"""Validate the exact engine against large Monte Carlo runs."""
import math
import sys
import time

import torch as th
from alg_zoo import handcrafted_2nd_argmax, zoo_2nd_argmax, zoo_argmedian
from alg_zoo.tasks import task_2nd_argmax, task_argmedian

from engine import RNN, exact_accuracy


def mc(model, n, task, N=2**23, seed=0):
    g = th.Generator().manual_seed(seed)
    x = th.randn(N, n, generator=g, dtype=th.float64)
    with th.no_grad():
        a = (model.double()(x).argmax(-1) == task(x)).double().mean().item()
    return a, math.sqrt(a * (1 - a) / N)


cases = [
    ("hand2", lambda: handcrafted_2nd_argmax(2), 2, "2nd_argmax", task_2nd_argmax),
    ("M2,2", lambda: zoo_2nd_argmax(2, 2), 2, "2nd_argmax", task_2nd_argmax),
    ("M4,3", lambda: zoo_2nd_argmax(4, 3), 3, "2nd_argmax", task_2nd_argmax),
    ("M3,3", lambda: zoo_2nd_argmax(3, 3), 3, "2nd_argmax", task_2nd_argmax),
    ("M4,4", lambda: zoo_2nd_argmax(4, 4), 4, "2nd_argmax", task_2nd_argmax),
    ("M8,4", lambda: zoo_2nd_argmax(8, 4), 4, "2nd_argmax", task_2nd_argmax),
    ("M16,4", lambda: zoo_2nd_argmax(16, 4), 4, "2nd_argmax", task_2nd_argmax),
    ("M8,5", lambda: zoo_2nd_argmax(8, 5), 5, "2nd_argmax", task_2nd_argmax),
    ("AM8,5", lambda: zoo_argmedian(8, 5), 5, "argmedian", task_argmedian),
    ("M16,5", lambda: zoo_2nd_argmax(16, 5), 5, "2nd_argmax", task_2nd_argmax),
]
names = sys.argv[1:] or [c[0] for c in cases[:6]]
for name, make, n, tname, task in cases:
    if name not in names:
        continue
    model = make()
    rnn = RNN.from_torch(model)
    t0 = time.time()
    acc, fl = exact_accuracy(rnn, tname)
    t1 = time.time()
    m, se = mc(model, n, task)
    z = (acc - m) / se if se > 0 else float("nan")
    print(
        f"{name}: exact={acc:.8f} mc={m:.6f}±{se:.6f} z={z:+.2f} time={t1-t0:.1f}s "
        f"flops={fl.total:.3g} (rnn {fl.rnn:.2g} split {fl.split:.2g} meas {fl.measure:.2g})"
        f" = {fl.total/rnn.forward_flops():.3g} fwd; {fl.counts}",
        flush=True,
    )
