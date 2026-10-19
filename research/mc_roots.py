"""Per-root Monte Carlo check of the exact engine.

The exact accuracy is a sum of 2 n! independent root-cone contributions (results/roots/). This
samples N Gaussian inputs (an RNG not used anywhere else), assigns each to its root cone
(ordering of the inputs and sign of x_0), and compares, root by root, the fraction of samples
that are in the root and classified correctly (the target comes from alg_zoo.tasks) with the
exact contribution, with a Pearson chi-square test over the roots whose expected count is at
least 5 plus one remainder cell (roots on which the model is wrong everywhere are checked
separately: no sample in them may be classified correctly). This is far more sensitive than
one aggregate z-score: an error confined to a few roots cannot hide behind the others.

Usage: python mc_roots.py <family:d,n> [log2 N]   (default N = 2^26; writes results/mc_roots.json)
"""

import itertools
import json
import math
import os
import sys

import numpy as np
import torch as th
from scipy import stats

from evaluate import load
from exact_parallel import checkpoint_path, read_jsonl


def per_root_mc(spec, N, chunk=2**20):
    rnn, (task, task_fn) = load(spec)
    n = rnn.n
    # root index = 2 * (index of the ordering among itertools.permutations) + (x_0 < 0)
    code = np.full(n**n, -1)
    for i, perm in enumerate(itertools.permutations(range(n))):
        code[sum(p * n**j for j, p in enumerate(perm))] = i
    counts = np.zeros(2 * math.factorial(n), np.int64)
    rng = np.random.Generator(np.random.SFC64([4242, n, rnn.d, int(spec.startswith("AM"))]))
    for start in range(0, N, chunk):
        x = rng.standard_normal((min(chunk, N - start), n))
        ok = rnn.forward(x).argmax(-1) == task_fn(th.from_numpy(x)).numpy()
        order = np.argsort(-x, axis=1)  # order[:, i] = position with rank i (0 = largest)
        k = 2 * code[(order * n ** np.arange(n)).sum(1)] + (x[:, 0] < 0)
        counts += np.bincount(k[ok], minlength=len(counts))
    return counts


if __name__ == "__main__":
    spec = sys.argv[1]
    N = 2 ** int(sys.argv[2]) if len(sys.argv) > 2 else 2**26
    fam, rest = spec.split(":")
    d, n = map(int, rest.split(","))
    rows = {r["root"]: r for r in read_jsonl(checkpoint_path(fam, d, n))}
    exact = np.array([rows[k]["contrib"] for k in range(2 * math.factorial(n))])
    counts = per_root_mc(spec, N)
    # roots on which the model is wrong everywhere have contribution 0 (up to rounding): no
    # sample in them may be classified correctly
    live = exact > 1e-14
    zero_root_hits = int(counts[~live].sum())
    # Pearson chi-square over the roots with an expected count of at least 5, plus one remainder
    # cell (incorrect samples and the samples correct in all other roots); dof = cells - 1
    E = N * exact
    big = live & (E >= 5)
    O_cells = np.append(counts[big], N - counts[big].sum())
    E_cells = np.append(E[big], N - E[big].sum())
    chi2 = float(((O_cells - E_cells) ** 2 / E_cells).sum())
    z = (counts[big] - E[big]) / np.sqrt(E[big] * (1 - exact[big]))
    out = dict(spec=spec, N=N, roots=int(big.sum()), pooled_roots=int((live & ~big).sum()),
               zero_roots=int((~live).sum()), zero_root_hits=zero_root_hits, chi2=chi2,
               dof=len(O_cells) - 1, p=float(stats.chi2.sf(chi2, len(O_cells) - 1)),
               max_abs_z=float(np.abs(z).max()), worst_root=int(np.where(big)[0][np.abs(z).argmax()]),
               min_expected=float(E[big].min()), exact=float(exact.sum()), mc=float(counts.sum() / N),
               z_total=float((exact.sum() - counts.sum() / N) / math.sqrt(exact.sum() * (1 - exact.sum()) / N)))
    print(json.dumps(out))
    path = "results/mc_roots.json"
    allr = json.load(open(path)) if os.path.exists(path) else {}
    allr[spec] = out
    json.dump(allr, open(path, "w"), indent=1)
