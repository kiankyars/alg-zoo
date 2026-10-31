# Pre-registration: held-out test of the cheap-measure ladder for 4-input models

Committed, with the code it describes, before any model of the test below was evaluated.
`evaluate.py`, with `REQUIRE_CLEAN` set by `run_test_cheap4.sh`, refuses to evaluate when anything
in the repository other than `research/results` is uncommitted, and stamps the commit into every
result row; the run script stops if it refuses.

## What is tested

`evaluate.py` rule `cheap4`, for 4-input models: the published `lguard` ladder (`engine.py`) with
every Gaussian measure computed by a cheap rule, `CHEAP = cheap:1:1.0` (degree-3 slice cubature
after bisecting cones to at most 1 radian, `engine.measure_simplices`), followed by two near-exact
rungs: `exact_cheap` (exact refinement measured with `CHEAP`) and `exact_fine` (exact refinement
measured with `FINE = cheap:2:1.0`, degree-5 cubature), then the exact rung (the reference engine,
whose value is the truth). In the fair rule, a rung that comes before a near-exact rung but costs
at least as much is never used and is dropped (`summarize._drop_dominated`).

Why: exact computation of a 4-input model costs about 10⁶ forward passes, 80-98% of it in Gaussian
measures (Plackett's formula and degree-7 cubature). Exact refinement itself costs 2-20%.

## How it was designed

* A design study (development seeds 5000-5015; `M_{8,4}`, `M_{16,4}`, `M_{32,4}`) measured the
  accuracy and cost of several cheap rules for the measures of exact refinement, and found the
  first-order sliver laws, localised refinement and other hybrids not competitive.
* Development seeds 5400-5415 (16 models per cell, the six cells below, `results/kdev4`): the
  `cheap4` ladder, first with `exact_cheap` only, then with `exact_fine` added, because
  `exact_cheap`'s 0.3% bias lost to sampling just below the exact cost of `M_{8,4}` at
  sigma = 0.01. The rows in `results/kdev4/*_cheap4.jsonl` were regenerated with the committed
  rule. The thresholds b0 below were set from these runs (1.1 x the mean cost of `exact_cheap`).
* A pre-freeze review used dev seeds 5500-5599. Seeds 7000-7007 were used once for `M_{16,4}` with
  the `lguard` ladder (`PREREG_lean.md`). No 4-input model has been evaluated on seeds 8000-8099.

## Data

* Models: `M_{8,4}` and `M_{16,4}`, sigma in {0.002, 0.005, 0.01} (6 cells).
* Noise: `evaluate.noisy(base, sigma, seed, keyed=True)`, seeds 8000-8099 (100 models per cell).
* Both ladders on every model (`run_test_cheap4.sh`, tag `test8000`). Truth: `engine.exact_accuracy`.
* No exclusions; a model whose run crashes is rerun, not dropped. All result files, the validation
  and the analysis are published whatever the outcome.
* Validation of the truth by plain Monte Carlo (2^19 samples per model, `mc_validate.py`). A cell
  is flagged if its chi-square p-value is below 0.001 or any of its models has |z| > 4; the flagged
  models are then rechecked with 2^24 samples, and a confirmed discrepancy (|z| > 4 again) voids
  that cell.

## Analysis (`analyze_test4.py`, unchanged after this commit)

* **Primary**: for each cell, H0 "the cheap4 ladder's worst fair ratio over budgets B >= b0 is
  >= 1", with b0 = 6.3 x 10⁴ forward passes for `M_{8,4}` and 1.35 x 10⁵ for `M_{16,4}` (about 5% and
  11% of their exact cost). p = share of 10,000 paired bootstrap replicates (seed 0) with a ratio
  >= 1; Holm at level 0.05 over the 6 cells. For B >= b0 the fair rule uses only `exact_cheap`,
  `exact_fine` and `exact` (b0 is above `exact_cheap`'s cost, and the near-exact rungs come last), so
  no `roots_k` or `any_k` rung enters: this is a held-out check that exact refinement with the two
  cheap cubatures beats sampling from b0 on, i.e. that their bias stays below sampling noise. On
  the development seeds the margin is large (see below), so a rejection is expected; what it
  establishes is the budget, b0, from which a mechanistic estimate beats sampling at every budget,
  against the exact cost before.
* Secondary, with 90% bootstrap intervals: each ladder's worst fair ratio over all budgets B >= 1
  (this one does involve the anytime rungs); their paired ratio, with Holm over the 6 cells for
  "cheap4 has the smaller worst ratio"; and each ladder's crossover budget B* (`summarize.crossover`),
  from which it beats sampling at every budget.

## Expectations stated in advance (from the development seeds)

From `analyze_test4.py results/kdev4` (16 models per cell, 2,000 bootstrap replicates):

* Primary: H0 rejected in all 6 cells. The worst ratio over B >= b0 is 0.017 / 0.040 / 0.074 for
  `M_{8,4}` and 0.014 / 0.032 / 0.062 for `M_{16,4}` (sigma 0.002 / 0.005 / 0.01), with 90% intervals
  below 0.09; a cell would fail only if the near-exact rungs' errors grew about 4-8 times in RMS.
* The worst ratio over all budgets stays far above 1 for both ladders (32-177 for cheap4, 755-3,850
  for lguard), so this does not solve the 4-input problem at intermediate budgets; cheap4's is about
  20 times smaller (paired ratio 0.040-0.053).
* Crossover B*: 4.3-5.3 x 10⁴ forward passes for `M_{8,4}` and 1.16-1.21 x 10⁵ for `M_{16,4}`; for
  lguard it is the exact cost (1.06-1.28 x 10⁶).
