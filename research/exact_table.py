"""Exact (sampling-free) accuracies of every AlgZoo RNN classifier with seq_len <= 5.

For each model: accuracy from the exact engine, its change when all Gaussian measures are recomputed
at higher order (Plackett with 28 instead of 20 nodes per panel, slice cubature of degree 9
instead of 7) as a numerical-precision check, compute cost in forward-pass equivalents, and a
Monte Carlo check with 2^24 samples (z-score).

Each accuracy is computed root cone by root cone in a process pool (exact_parallel.py), with
per-root checkpoints in results/roots/, so the large 5-input models can be resumed.

Usage: python exact_table.py <n> [<n> ...]   (writes results/exact_table_n<ns>.json)
"""

import json
import math
import os
import sys

import torch as th
from alg_zoo import zoo_2nd_argmax, zoo_argmedian
from alg_zoo.tasks import task_2nd_argmax, task_argmedian

from engine import RNN
from exact_parallel import checkpoint_path, exact_accuracy_parallel, read_jsonl

NS = [int(a) for a in sys.argv[1:]] or [2, 3, 4, 5]
SPECS = [("M", d, n) for n in (2, 3, 4, 5) for d in (2, 3, 4, 5, 6, 8, 16, 32) if n in NS]
# argmedian with n = 3 is the same task as 2nd argmax and the zoo models have identical weights
SPECS += [("AM", d, 5) for d in (2, 4, 8, 16, 32) if 5 in NS]
WORKERS = int(os.environ.get("WORKERS", 9))  # processes per model


def higher_order(fam, d, n, rnn, task):
    """The accuracy with every Gaussian measure recomputed at higher order (a precision check)."""
    return exact_accuracy_parallel(rnn, task, checkpoint_path(fam, d, n, s=4, n_quad=28), WORKERS,
                                   s=4, n_quad=28)[0]


def run(spec):
    fam, d, n = spec
    model = (zoo_2nd_argmax if fam == "M" else zoo_argmedian)(d, n)
    task, task_fn = ("2nd_argmax", task_2nd_argmax) if fam == "M" else ("argmedian", task_argmedian)
    rnn = RNN.from_torch(model)
    os.makedirs(os.path.dirname(checkpoint_path(fam, d, n)), exist_ok=True)
    acc, fl, roots = exact_accuracy_parallel(rnn, task, checkpoint_path(fam, d, n), WORKERS,
                                             return_rows=True)
    # sum over roots of each root's time in the 9-process pool (roughly single-core time)
    seconds = sum(r["seconds"] for r in roots)
    acc2 = higher_order(fam, d, n, rnn, task)
    g = th.Generator().manual_seed(7)
    N, hits = 2**24, 0
    for _ in range(16):
        x = th.randn(N // 16, n, generator=g, dtype=th.float64)
        hits += int((rnn.forward(x.numpy()).argmax(-1) == task_fn(x).numpy()).sum())
    mc = hits / N
    se = math.sqrt(max(mc * (1 - mc), 1e-12) / N)
    return dict(family=fam, d=d, n=n, params=sum(p.numel() for p in model.parameters()),
                exact=acc, exact_higher_order=acc2,
                measure_disagreement=None if acc2 is None else abs(acc - acc2),
                cost_fwd=fl.total / rnn.forward_flops(), seconds=seconds, leaves=fl.counts["leaves"],
                mc=mc, mc_se=se, z=(acc - mc) / se if se > 0 else 0.0)


def dedupe(rows):
    """One row per model, preferring one that has the higher-order check, then the later one."""
    best = {}
    for r in rows:
        key = (r["family"], r["d"], r["n"])
        if key not in best or r["exact_higher_order"] is not None or best[key]["exact_higher_order"] is None:
            best[key] = r
    return list(best.values())


if __name__ == "__main__":
    import fcntl

    # resumable: each finished model is appended to a JSONL file; finished models are skipped
    tag = "".join(map(str, NS))
    part = f"results/exact_table_n{tag}.partial.jsonl"
    lock = open(part + ".lock", "w")
    try:  # one run per partial file at a time
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another exact_table.py run holds {part}.lock")
    done = dedupe(read_jsonl(part)) if os.path.exists(part) else []
    seen = {(r["family"], r["d"], r["n"]) for r in done}
    todo = [s for s in SPECS if s not in seen]
    with open(part, "a") as fh:
        for spec in todo:
            r = run(spec)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            done.append(r)
    # rows saved before the precision check was run for every model: fill it in
    missing = [r for r in done if r["exact_higher_order"] is None]
    for r in missing:
        rnn = RNN.from_torch((zoo_2nd_argmax if r["family"] == "M" else zoo_argmedian)(r["d"], r["n"]))
        task = "2nd_argmax" if r["family"] == "M" else "argmedian"
        r["exact_higher_order"] = higher_order(r["family"], r["d"], r["n"], rnn, task)
        r["measure_disagreement"] = abs(r["exact"] - r["exact_higher_order"])
        with open(part + ".tmp", "w") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in done)
        os.replace(part + ".tmp", part)
    order = {s: i for i, s in enumerate(SPECS)}
    rows = sorted(done, key=lambda r: order[(r["family"], r["d"], r["n"])])
    json.dump(rows, open(f"results/exact_table_n{tag}.json", "w"), indent=1)
    print(f"{'model':12s} {'params':>6s} {'exact accuracy':>16s} {'higher-order diff':>17s} {'cost (fwd)':>11s} {'time':>7s} {'MC (2^24)':>10s} {'z':>6s}")
    for r in rows:
        name = f"{r['family']}{r['d']},{r['n']}"
        print(f"{name:12s} {r['params']:6d} {r['exact']:16.10f} {'-' if r['measure_disagreement'] is None else format(r['measure_disagreement'], '.1e'):>16s} {r['cost_fwd']:11.3g} {r['seconds']:6.1f}s {r['mc']:10.6f} {r['z']:+6.2f}")
