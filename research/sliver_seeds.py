"""The tie-sliver laws on every training seed (0-4) of the larger zoo models, not just the best-of-5
seed in the zoo: the first-order law, the order-truncated law, the first-order law with the
bisector cap and the exact sliver law. For 4-input models the laws and the error are exact
(sliver.py, engine.py). For 5-input models, whose exact error costs ~1 CPU-hour per model, the
error and the laws are Monte Carlo estimates (sliver_mc.py; the exact laws agree with their
Monte Carlo estimates, see test_sliver.py).

Usage: python sliver_seeds.py   (writes results/sliver_seeds.json; resumable)
"""

import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
from alg_zoo import zoo_2nd_argmax, zoo_argmedian

from engine import RNN, exact_accuracy, target_rank
from sliver import sliver_error
from sliver_mc import decompose, law, widths

SPECS = [(fam, d, 4, s) for fam, d in (("M", 8), ("M", 16), ("M", 32)) for s in range(5)]
SPECS += [(fam, d, 5, s) for fam, d in (("M", 16), ("M", 32), ("AM", 16), ("AM", 32)) for s in range(5)]
PATH = "results/sliver_seeds.json"


def run(spec):
    fam, d, n, seed = spec
    rnn = RNN.from_torch((zoo_2nd_argmax if fam == "M" else zoo_argmedian)(d, n, seed=seed))
    task = "2nd_argmax" if fam == "M" else "argmedian"
    row = dict(family=fam, d=d, n=n, seed=seed)
    if n <= 4:
        acc, _ = exact_accuracy(rnn, task)
        row.update(error=1 - acc, error_se=0.0, method="exact")
        row["first_order"] = sliver_error(rnn, task, cap="none")[0]
        row["truncated"] = sliver_error(rnn, task, cap="chamber")[0]
        row["bisector"] = sliver_error(rnn, task, cap="bisector")[0]
        row["exact_sliver"] = sliver_error(rnn, task, cap="bisector", width="exact")[0]
    else:
        rng = np.random.default_rng(seed)
        N, wrong, r = 1 << 24, 0, target_rank(task, n)
        for _ in range(16):
            X = rng.standard_normal((N // 16, n))
            wrong += int((rnn.forward(X).argmax(1) != np.argsort(-X, 1)[:, r]).sum())
        e = wrong / N
        (f, fse), (t, tse), (bi, bse) = law(rnn, task, 1 << 20, rng, bisector=True)
        ws, wse = widths(rnn, task, 1 << 15, rng)["bisector"]
        row.update(error=e, error_se=math.sqrt(e * (1 - e) / N), method="monte carlo",
                   first_order=f, first_order_se=fse, truncated=t, truncated_se=tse,
                   bisector=bi, bisector_se=bse, exact_sliver=ws, exact_sliver_se=wse)
    return add_decomposition(row, rnn, task)


def add_new_laws(row):
    """The bisector-capped first-order law and the exact sliver law for rows saved before them."""
    if "exact_sliver" in row:
        return row
    rnn = RNN.from_torch((zoo_2nd_argmax if row["family"] == "M" else zoo_argmedian)(
        row["d"], row["n"], seed=row["seed"]))
    task = "2nd_argmax" if row["family"] == "M" else "argmedian"
    row = dict(row)
    if row["n"] <= 4:
        row["bisector"] = sliver_error(rnn, task, cap="bisector")[0]
        row["exact_sliver"] = sliver_error(rnn, task, cap="bisector", width="exact")[0]
    else:
        rng = np.random.default_rng(1000 + row["seed"])
        _, _, (bi, bse) = law(rnn, task, 1 << 20, rng, bisector=True)
        ws, wse = widths(rnn, task, 1 << 15, rng)["bisector"]
        row.update(bisector=bi, bisector_se=bse, exact_sliver=ws, exact_sliver_se=wse)
    return row


def add_decomposition(row, rnn=None, task=None):
    """Monte Carlo shares of the wrong inputs in one / two / no slivers (as in sliver_mc.py)."""
    if "two" in row:
        return row
    if rnn is None:
        rnn = RNN.from_torch((zoo_2nd_argmax if row["family"] == "M" else zoo_argmedian)(
            row["d"], row["n"], seed=row["seed"]))
        task = "2nd_argmax" if row["family"] == "M" else "argmedian"
    (one, two, none), err = decompose(rnn, task, 1 << 22, np.random.default_rng(100 + row["seed"]))
    row.update(one=one / err, two=two / err, none=none / err)
    return row


if __name__ == "__main__":
    rows = json.load(open(PATH)) if os.path.exists(PATH) else []
    done = {(r["family"], r["d"], r["n"], r["seed"]) for r in rows}
    todo = [s for s in SPECS if s not in done]
    with Pool(int(os.environ.get("WORKERS", 4))) as pool:
        rows = list(pool.imap(add_decomposition, rows))  # rows saved before the decomposition
        json.dump(rows, open(PATH, "w"), indent=1)
        out = []
        for r in pool.imap(add_new_laws, rows):  # rows saved before the new laws
            out.append(r)
            json.dump(out + rows[len(out):], open(PATH, "w"), indent=1)
            print(f"{r['family']}{r['d']},{r['n']} seed {r['seed']}: bisector {(r['bisector'] - r['error']) / r['error']:+.2%} "
                  f"exact sliver {(r['exact_sliver'] - r['error']) / r['error']:+.2%}", flush=True)
        rows = out
        for r in pool.imap_unordered(run, todo):
            rows.append(r)
            json.dump(rows, open(PATH, "w"), indent=1)
            print(f"{r['family']}{r['d']},{r['n']} seed {r['seed']}: error {r['error']:.4e} ({r['method']}) "
                  f"first-order {(r['first_order'] - r['error']) / r['error']:+.1%} "
                  f"truncated {(r['truncated'] - r['error']) / r['error']:+.1%}", flush=True)
    sys.exit()
