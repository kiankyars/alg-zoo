"""Independent Monte Carlo validation of the held-out ground truth (results/test).

The ground truth in results/test is the exact engine, which the estimator ladder shares code
with, so the exact rung's zero MSE holds by construction. This checks the engine against plain
sampling instead: for every noisy model in the given test files, 2^19 fresh Gaussian inputs (a
different RNG from everything else), z = (exact - MC) / SE, and a chi-square test over models.

Usage: python mc_validate.py results/test/M_32_3_0.01_lguard.jsonl ...   (writes results/mc_validation.json)
"""

import json
import math
import sys
from multiprocessing import Pool

import numpy as np
from alg_zoo.tasks import task_2nd_argmax
from scipy import stats

from evaluate import load, noisy

N = 2**19


def check(args):
    spec, sigma, seed, truth = args
    base, _ = load(spec)
    rnn = noisy(base, sigma, seed, keyed=True)
    rng = np.random.Generator(np.random.SFC64([seed, 7919]))
    x = rng.standard_normal((N, rnn.n))
    pred = rnn.forward(x).argmax(-1)
    target = np.argsort(x, axis=-1)[:, -2]
    p = float((pred == target).mean())
    se = math.sqrt(max(p * (1 - p), 1e-12) / N)
    return seed, truth, p, (truth - p) / se


if __name__ == "__main__":
    import os

    dest = "results/mc_validation.json"
    out = json.load(open(dest)) if os.path.exists(dest) else {}
    for path in sys.argv[1:]:
        name = path.split("/")[-1]
        if name in out:
            continue
        d = int(name.split("_")[1])
        sigma = float(name.split("_")[3])
        rows = [json.loads(l) for l in open(path)]
        jobs = [(f"M:{d},3", sigma, r["seed"], r["truth"]) for r in rows]
        with Pool(9) as pool:
            res = pool.map(check, jobs)
        z = np.array([r[3] for r in res])
        chi2 = float((z**2).sum())
        out[name] = dict(models=len(z), samples_per_model=N, max_abs_z=float(np.abs(z).max()),
                         mean_z=float(z.mean()), chi2=chi2, dof=len(z),
                         chi2_p=float(stats.chi2.sf(chi2, len(z))), z=z.tolist())
        print(f"{name}: {len(z)} models, max|z| {np.abs(z).max():.2f}, mean z {z.mean():+.3f}, "
              f"chi2 {chi2:.1f} on {len(z)} df (p = {out[name]['chi2_p']:.2f})", flush=True)
        json.dump(out, open(dest, "w"), indent=1)  # saved after every file (resumable)
