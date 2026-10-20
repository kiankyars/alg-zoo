"""Tie-sliver laws vs the exact error, for the trained (un-noised) zoo models: the first-order law,
the order-truncated law (chamber cap), the first-order law with the bisector cap, and the exact
sliver law (bisector cap, exact widths along the normals; see sliver.sliver_error). Exact errors
are taken from results/exact_table_n*.json when available. The exact sliver law is not run for
the models in NO_EXACT, where it would take ~100 CPU-hours (sliver_mc.py widths estimates it).

Each (model, law, tie pair) is a separate job in a process pool, so the large 5-input models are
spread over all workers; finished jobs are appended to results/sliver_table_n<ns>.jobs.jsonl,
so runs are resumable.
Usage: python sliver_table.py <n> [<n> ...]   (writes results/sliver_table_n<ns>.json)"""

import functools
import glob
import itertools
import json
import os
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
WORKERS = int(os.environ.get("WORKERS", 9))
LAWS = {"sliver": ("none", "first"), "truncated": ("chamber", "first"),  # name -> (cap, width)
        "bisector": ("bisector", "first"), "exact": ("bisector", "exact")}
NO_EXACT = {("M", 32, 5), ("AM", 32, 5)}


@functools.lru_cache(maxsize=None)
def load(spec):
    fam, d, n = spec
    model = (zoo_2nd_argmax if fam == "M" else zoo_argmedian)(d, n)
    return RNN.from_torch(model), "2nd_argmax" if fam == "M" else "argmedian"


def exact(spec):
    if spec in KNOWN:
        return 1 - KNOWN[spec]["exact"], KNOWN[spec]["cost_fwd"]
    rnn, task = load(spec)
    acc, fle = exact_accuracy(rnn, task)
    return 1 - acc, fle.total / rnn.forward_flops()


def job(args):
    spec, law, pair = args
    rnn, task = load(spec)
    t0 = time.time()
    cap, width = LAWS[law]
    e, fl = sliver_error(rnn, task, cap=cap, width=width, pairs={tuple(pair)})
    return dict(spec=list(spec), law=law, pair=list(pair), error=e, cost=fl.total / rnn.forward_flops(),
                seconds=time.time() - t0)


if __name__ == "__main__":
    tag = "".join(map(str, NS))
    jobs_path = f"results/sliver_table_n{tag}.jobs.jsonl"
    done = {}
    if os.path.exists(jobs_path):
        for line in open(jobs_path):
            r = json.loads(line)
            done[(tuple(r["spec"]), r["law"], tuple(r["pair"]))] = r
    todo = [(s, law, pair) for s in SPECS for law in LAWS for pair in itertools.combinations(range(s[2]), 2)
            if (s, law, pair) not in done and not (law == "exact" and s in NO_EXACT)]
    todo.sort(key=lambda j: (j[0][2], j[0][1], j[1] == "exact"))  # smallest models first
    with Pool(WORKERS) as pool, open(jobs_path, "a") as fh:
        for r in pool.imap_unordered(job, todo):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            done[(tuple(r["spec"]), r["law"], tuple(r["pair"]))] = r
    rows = []
    for s in SPECS:
        fam, d, n = s
        ex, exact_cost = exact(s)
        row = dict(family=fam, d=d, n=n, exact_error=ex, exact_cost=exact_cost)
        for law in LAWS:
            if law == "exact" and s in NO_EXACT:
                continue
            parts = [done[(s, law, pair)] for pair in itertools.combinations(range(n), 2)]
            e = sum(p["error"] for p in parts)
            row.update({f"{law}_error": e, f"{law}_rel_diff": (e - ex) / ex,
                        f"{law}_cost": sum(p["cost"] for p in parts),
                        f"{law}_seconds": sum(p["seconds"] for p in parts)})
        row["rel_diff"] = row["sliver_rel_diff"]  # (the name used by earlier versions of this table)
        rows.append(row)
    json.dump(rows, open(f"results/sliver_table_n{tag}.json", "w"), indent=1)
    print(f"{'model':8s} {'exact error':>12s} " + " ".join(f"{law:>18s}" for law in LAWS))
    for r in rows:
        print(f"{r['family']}{r['d']},{r['n']:<4d} {r['exact_error']:12.4e} " + " ".join(
            f"{r[law + '_error']:10.4e} {r[law + '_rel_diff']:+7.2%}" if law + "_error" in r else f"{'-':>18s}"
            for law in LAWS))
