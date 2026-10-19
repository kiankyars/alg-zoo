"""exact_accuracy split over root cones: the 2 n! roots are independent, so they run in a process
pool, and each finished root is appended to a JSONL checkpoint, so an interrupted run resumes
where it stopped. The result equals exact_accuracy's up to floating-point summation order.

Usage: python exact_parallel.py <family:d,n> [workers]   (e.g. M:16,5; family M or AM)
Writes results/roots/<family>_<d>_<n>_s3_q20.jsonl (one line per root).
"""

import json
import math
import os
import sys
import time
from multiprocessing import Pool

from engine import Flops, root_accuracy


def read_jsonl(path):
    """Rows of a JSONL checkpoint, skipping a line truncated by an interrupted write."""
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _work(args):
    rnn, task, k, kw = args
    t0, c0 = time.time(), time.process_time()
    a, fl = root_accuracy(rnn, task, k, **kw)
    return dict(root=k, contrib=a, rnn=fl.rnn, split=fl.split, measure=fl.measure,
                seconds=time.time() - t0, cpu_seconds=time.process_time() - c0, **fl.counts)


def exact_accuracy_parallel(rnn, task, checkpoint=None, workers=9, measure="slice", s=3, n_quad=20,
                            log=None, return_rows=False):
    """Same as engine.exact_accuracy (returns (accuracy, Flops)), computed root by root in a
    process pool. With `checkpoint` (a JSONL path) finished roots are saved and skipped on
    resume; the caller must make sure the path identifies the model and the settings. With
    return_rows=True also returns the per-root rows used (one per root)."""
    kw = dict(measure=measure, s=s, n_quad=n_quad)
    n_roots = 2 * math.factorial(rnn.n)
    done = {}
    if checkpoint and os.path.exists(checkpoint):
        for r in read_jsonl(checkpoint):
            done[r["root"]] = r
    todo = [(rnn, task, k, kw) for k in range(n_roots) if k not in done]
    if todo:
        if checkpoint:
            os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
        fh = open(checkpoint, "a") if checkpoint else None
        if fh and fh.tell() > 0 and not open(checkpoint, "rb").read()[-1:] == b"\n":
            fh.write("\n")
        with Pool(min(workers, len(todo))) as pool:
            for r in pool.imap_unordered(_work, todo):
                done[r["root"]] = r
                if fh:
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                if log:
                    log(f"{len(done)}/{n_roots} roots")
        if fh:
            fh.close()
    rows = [done[k] for k in range(n_roots)]
    fl = Flops()
    for r in rows:
        fl.add(Flops(rnn=r["rnn"], split=r["split"], measure=r["measure"],
                     counts={c: r[c] for c in ("splits", "leaves", "simplices")}))
    acc = math.fsum(r["contrib"] for r in rows)
    return (acc, fl, rows) if return_rows else (acc, fl)


def checkpoint_path(family, d, n, s=3, n_quad=20):
    return os.path.join("results", "roots", f"{family}_{d}_{n}_s{s}_q{n_quad}.jsonl")


if __name__ == "__main__":
    from evaluate import load

    spec = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    fam, rest = spec.split(":")
    d, n = map(int, rest.split(","))
    rnn, (task, _) = load(spec)
    path = checkpoint_path(fam, d, n)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    acc, fl = exact_accuracy_parallel(rnn, task, path, workers,
                                      log=lambda m: print(f"{m} ({time.time() - t0:.0f}s)", flush=True))
    print(f"{spec}: accuracy {acc:.12f}, cost {fl.total / rnn.forward_flops():.4g} fwd, {fl.counts}")
