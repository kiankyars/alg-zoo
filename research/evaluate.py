"""MSE-vs-compute evaluation of the mechanistic estimator ladder on noisy-weight ensembles.

ARC's suggestion: perturb the weights of a trained model enough to significantly change its
accuracy, and average the squared error of an estimator over the noisy models. Our instantiation:
each weight tensor gets sigma * RMS(tensor) * N(0, 1) noise, with the noise drawn from an RNG keyed
by (seed, sigma), so different noise levels use independent noise. Sampling with B forward passes
has MSE  mean_k p_k (1 - p_k) / B  (p_k = accuracy of noisy model k). Compute is measured in
forward-pass equivalents: estimator flops / RNN.forward_flops().

Usage: python evaluate.py <family:d,n> <sigma> <first_seed> <n_models> <rule> [tag]
(rule: an engine._finish rule such as lguard, or "lean" for the ladder of lean.py)
Writes results/<tag>/<spec>_<sigma>_<rule>.jsonl (one line per noisy model; resumable).
"""

import functools
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch as th
from alg_zoo import zoo_2nd_argmax, zoo_argmedian
from alg_zoo.tasks import task_2nd_argmax, task_argmedian

from engine import RNN, anytime_ladder, exact_accuracy
from lean import anytime_ladder_lean, exact_accuracy_lean

TASKS = {"M": ("2nd_argmax", task_2nd_argmax), "AM": ("argmedian", task_argmedian)}
SPLIT_GRID = [0, 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 256, 362, 512, 724,
              1024, 1448, 2048, 2896, 4096, 5793, 8192, 11585, 16384]
ROOT_GRID = [1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181]


@functools.lru_cache(maxsize=None)
def load(spec):
    """Base zoo model (cached per process: it is downloaded from GCS) and its task."""
    fam, rest = spec.split(":")
    d, n = map(int, rest.split(","))
    model = (zoo_2nd_argmax if fam == "M" else zoo_argmedian)(d, n)
    return RNN.from_torch(model), TASKS[fam]


def noisy(rnn, sigma, seed, keyed=True):
    """Perturb each weight tensor by sigma * (its RMS) * standard Gaussian noise. With keyed=True
    the noise RNG is keyed by (seed, sigma) so that different sigmas get independent noise."""
    rng = np.random.default_rng([seed, int(round(sigma * 1e5))] if keyed else seed)

    def pert(W):
        return W + sigma * np.sqrt((W**2).mean()) * rng.standard_normal(W.shape)

    return RNN(Wi=pert(rnn.Wi), Wh=pert(rnn.Wh), Wo=pert(rnn.Wo), n=rnn.n)


def run_one(args):
    spec, sigma, seed, rule, keyed = args
    base, (task, _) = load(spec)
    rnn = noisy(base, sigma, seed, keyed)
    fwd = rnn.forward_flops()
    t0 = time.time()
    truth, fl = exact_accuracy(rnn, task)
    if rule == "lean":  # lean.py: its own exact rung (same answer to ~1e-10, cheaper bookkeeping)
        acc, fl_lean = exact_accuracy_lean(rnn, task)
        rungs = {"const": (1.0, 0.0), "exact": (acc, fl_lean.total / fwd)}
        ladder = anytime_ladder_lean(rnn, task, SPLIT_GRID, ROOT_GRID)
    else:
        rungs = {"const": (1.0, 0.0), "exact": (truth, fl.total / fwd)}
        ladder = anytime_ladder(rnn, task, SPLIT_GRID, ROOT_GRID, rule=rule)
    for name, (est, flops) in ladder.items():
        rungs[name] = (est, flops / fwd)
    return {"seed": seed, "truth": truth, "rungs": rungs, "seconds": time.time() - t0}


def mc_check(spec, sigma, seed, keyed=True, N=2**22):
    base, (task, task_fn) = load(spec)
    rnn = noisy(base, sigma, seed, keyed)
    g = th.Generator().manual_seed(10**6 + seed)
    x = th.randn(N, rnn.n, generator=g, dtype=th.float64)
    a = float((rnn.forward(x.numpy()).argmax(-1) == task_fn(x).numpy()).mean())
    return a, math.sqrt(a * (1 - a) / N)


def path_for(spec, sigma, rule, tag):
    name = f"{spec.replace(':', '_').replace(',', '_')}_{sigma}_{rule}.jsonl"
    return os.path.join("results", tag, name)


if __name__ == "__main__":
    spec, sigma, first, K, rule = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    tag = sys.argv[6] if len(sys.argv) > 6 else "test"
    keyed = not tag.startswith("dev")  # dev sets reuse the original seed-only noise (as in the review)
    path = path_for(spec, sigma, rule, tag)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {json.loads(l)["seed"] for l in open(path)} if os.path.exists(path) else set()
    todo = [(spec, sigma, s, rule, keyed) for s in range(first, first + K) if s not in done]
    with Pool(int(os.environ.get("WORKERS", 9))) as pool, open(path, "a") as fh:
        for r in pool.imap_unordered(run_one, todo):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    print(f"{spec} sigma={sigma} rule={rule} tag={tag}: {K} models in {path}")
