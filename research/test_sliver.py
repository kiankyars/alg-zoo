"""Compare the first-order tie-sliver error estimate with the exact error."""
import sys
import time

from alg_zoo import zoo_2nd_argmax, zoo_argmedian

from engine import RNN, exact_accuracy
from sliver import sliver_error

for name in sys.argv[1:]:
    fam, rest = name.split(":")
    d, n = map(int, rest.split(","))
    model = zoo_2nd_argmax(d, n) if fam == "M" else zoo_argmedian(d, n)
    task = "2nd_argmax" if fam == "M" else "argmedian"
    rnn = RNN.from_torch(model)
    fwd = rnn.forward_flops()
    t0 = time.time()
    acc, fle = exact_accuracy(rnn, task)
    t1 = time.time()
    e, fls, det = sliver_error(rnn, task, return_details=True)
    t2 = time.time()
    ex = 1 - acc
    print(f"{name}: exact err {ex:.6e} ({fle.total / fwd:.3g} fwd, {t1 - t0:.1f}s) | sliver {e:.6e} "
          f"rel err {(e - ex) / ex:+.3%} abs {e - ex:+.2e} ({fls.total / fwd:.3g} fwd, {t2 - t1:.1f}s) {det}", flush=True)
