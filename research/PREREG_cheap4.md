# Pre-registration: held-out test of the cheap-measure ladder for 4-input models

Committed, with the code it describes, before any model of the test below was evaluated.
`evaluate.py` (with `REQUIRE_CLEAN`, set by `run_test_cheap4.sh`) refuses to evaluate uncommitted
code and stamps the commit into every result row; the run script stops if it refuses.

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

* A design study (development seeds 5000-5015, `M_{8,4}`, `M_{16,4}`, `M_{32,4}`) measured the
  accuracy and cost of several cheap rules for the measures of exact refinement, and found the
  first-order sliver laws, localised refinement and other hybrids not competitive.
* Development seeds 5400-5415 (16 models per cell, all six cells below): the `cheap4` ladder, first
  with `exact_cheap` only, then with `exact_fine` added because `exact_cheap`'s 0.3% bias lost to
  sampling just below the exact cost of `M_{8,4}` at sigma = 0.01. The thresholds b0 below were set
  from these runs (1.5 x the mean cost of `exact_cheap`).
* No 4-input model has been evaluated on seeds 8000-8099.

## Data

* Models: `M_{8,4}` and `M_{16,4}`, sigma in {0.002, 0.005, 0.01} (6 cells).
* Noise: `evaluate.noisy(base, sigma, seed, keyed=True)`, seeds 8000-8099 (100 models per cell).
* Both ladders on every model (`run_test_cheap4.sh`, tag `test8000`). Truth: `engine.exact_accuracy`,
  validated by plain Monte Carlo (2^19 samples per model, `mc_validate.py`) in all 6 cells.
* No exclusions; a model whose run crashes is rerun, not dropped. All result files, the validation
  and the analysis are published whatever the outcome.

## Analysis (`analyze_test4.py`, unchanged after this commit)

* **Primary**: for each cell, H0 "the cheap4 ladder's worst fair ratio over budgets B >= b0 is
  >= 1", with b0 = 8.5 x 10⁴ forward passes for `M_{8,4}` and 1.8 x 10⁵ for `M_{16,4}` (about 7% and
  15% of their exact cost). p = share of 10,000 paired bootstrap replicates (seed 0) with a ratio
  >= 1; Holm at level 0.05 over the 6 cells. A rejected H0 means: from b0 on, the ladder has lower
  MSE than sampling at every budget.
* Secondary, with 90% bootstrap intervals: each ladder's worst fair ratio over all budgets B >= 1;
  their paired ratio, with Holm over the 6 cells for "cheap4 has the smaller worst ratio"; and each
  ladder's crossover budget B* (`summarize.crossover`), from which it beats sampling at every budget.

## Expectations stated in advance (from the development seeds)

H0 rejected in all 6 cells (dev: worst ratio over B >= b0 of 0.0003-0.04). The worst ratio over all
budgets stays far above 1 for both ladders (dev: 32-177 for cheap4, 755-3,850 for lguard), so the
4-input problem at intermediate budgets is not solved by this; cheap4's is about 20x smaller.
Crossover B*: about 4-6 x 10⁴ forward passes for `M_{8,4}` and 1.2 x 10⁵ for `M_{16,4}`, against the
exact cost (about 1.1-1.3 x 10⁶) for lguard.
