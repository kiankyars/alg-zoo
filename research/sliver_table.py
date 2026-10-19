"""Tie-sliver law: first-order sliver estimate of the error vs the exact error, for the trained
(un-noised) zoo models. Exact errors are taken from results/exact_table_n*.json when available.
Usage: python sliver_table.py <n> [<n> ...]   (writes results/sliver_table_n<ns>.json)"""

import glob
import json
import sys
import time
from multiprocessing import Pool

from alg_zoo import zoo_2nd_argmax, zoo_argmedian

from engine import RNN, exact_accuracy
from sliver import sliver_error

# (argmedian with n = 3 is the same task as 2nd argmax, and the zoo models have identical weights)
SPECS = [("M", d, 2) for d in (2, 3, 4, 5, 6, 8, 16, 32)] + [("M", d, 3) for d in (3, 4, 5, 6, 8, 16, 32)]
SPECS += [("M", d, 4) for d in (6, 8, 16, 32)]
SPECS += [("M", d, 5) for d in (6, 8, 16, 32)] + [("AM", d, 5) for d in (8, 16, 32)]
NS = [int(a) for a in sys.argv[1:]] or [2, 3, 4, 5]
SPECS = [s for s in SPECS if s[2] in NS]
KNOWN = {(r["family"], r["d"], r["n"]): r for p in glob.glob("results/exact_table_n*.json")
         for r in json.load(open(p))}


def run(spec):
    fam, d, n = spec
    model = (zoo_2nd_argmax if fam == "M" else zoo_argmedian)(d, n)
    task = "2nd_argmax" if fam == "M" else "argmedian"
    rnn = RNN.from_torch(model)
    fwd = rnn.forward_flops()
    if spec in KNOWN:
        acc, exact_cost = KNOWN[spec]["exact"], KNOWN[spec]["cost_fwd"]
    else:
        acc, fle = exact_accuracy(rnn, task)
        exact_cost = fle.total / fwd
    t0 = time.time()
    e, fls = sliver_error(rnn, task)
    return dict(family=fam, d=d, n=n, exact_error=1 - acc, sliver_error=e,
                rel_diff=(e - (1 - acc)) / (1 - acc), exact_cost=exact_cost,
                sliver_cost=fls.total / fwd, sliver_seconds=time.time() - t0)


if __name__ == "__main__":
    # resumable: each finished model is appended to a JSONL file; finished models are skipped
    import os

    tag = "".join(map(str, NS))
    part = f"results/sliver_table_n{tag}.partial.jsonl"
    done = [json.loads(line) for line in open(part)] if os.path.exists(part) else []
    seen = {(r["family"], r["d"], r["n"]) for r in done}
    todo = [s for s in SPECS if s not in seen]
    with Pool(9) as pool, open(part, "a") as fh:
        for r in pool.imap_unordered(run, todo):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            done.append(r)
    order = {s: i for i, s in enumerate(SPECS)}
    rows = sorted(done, key=lambda r: order[(r["family"], r["d"], r["n"])])
    json.dump(rows, open(f"results/sliver_table_n{tag}.json", "w"), indent=1)
    print(f"{'model':8s} {'exact error':>12s} {'sliver law':>12s} {'rel diff':>9s} {'exact cost':>11s} {'sliver cost':>11s}")
    for r in rows:
        print(f"{r['family']}{r['d']},{r['n']:<4d} {r['exact_error']:12.4e} {r['sliver_error']:12.4e} "
              f"{r['rel_diff']:+9.1%} {r['exact_cost']:11.3g} {r['sliver_cost']:11.3g}")
