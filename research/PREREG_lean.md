# Pre-registration: held-out test of the lean ladder for 3-input models

Committed, with the code it describes, before any model of the test below was evaluated.
`run_test_lean.sh` refuses to run on uncommitted code and stamps the commit into every result row.

## What is tested

`lean.py` (`evaluate.py` rule `lean`): a second estimator ladder for 3-input models. Compared with
the published `lguard` ladder (`engine.py`), it

* finishes a cone exactly when its first conflict is at the last step and its tie linearisation
  claims more than 10% of the cone is wrong;
* linearises a cone that has both of its target's tie faces next to each face, on the two sides
  of their bisector;
* examines the root cones of its cheapest rungs (`roots_k`) in decreasing order of their
  closed-form Gaussian measure;
* uses cheaper bookkeeping for `n = 3` with the same estimates: Girard's formula for cone
  measures, linearisation from the point's pre-activations only, and step 0 of a root from the
  sign of `x_0`. The exact rung is exact refinement with the same bookkeeping, scored at the
  truth (it agrees with it to at most 3e-9).

## How it was designed (everything the test must be independent of)

* The published held-out test of `lguard` (noise seeds 1000-1199, `results/test/`) was seen by
  everyone involved; the cells it failed (`M_{4,3}`, `M_{6,3}`, `M_{8,3}` at all noise levels,
  `M_{5,3}` and `M_{16,3}` at sigma = 0.01) were the design target. That test is therefore not
  used to evaluate `lean`.
* Development noise seeds, all keyed by (seed, sigma): 5000-5131 (design and a first fresh
  confirmation on 5100-5131), 5200-5223, 5900-5915 and 5960-6007 (checks by the judges of the
  design), 5600-5799 and 5000-5031 (reviews of the port).
* Seeds 7000-7007 were used once, by a reviewer, for the `lguard` ladder on the 4-input `M_{16,4}`
  at sigma = 0.005. No 3-input model has been evaluated on seeds 7000-7199.
* Changes after the design and before this freeze, all from the review of the port: the exact
  rung is scored at the truth (as in the design; the port had used its own value, which differs
  by rounding and would have made `summarize.py` report an infinite ratio); `engine.split` now
  also charges a hyperplane test that leaves the cone on one side, and the normalisation, masks
  and new-ray norms of a split (this applies to both ladders and to the exact rungs); `lean.py`
  charges its tie-face tests, tie points and sign comparisons in full (including a tie-face test
  that `engine._tie_point` repeats). On 12 development models these raise the charged costs of
  the `lean` rungs by 7-15% (median 10%), of the `lguard` rungs by 2-9% (median 6%) and of the
  exact rungs by 10-13%; no estimate changes.

## Data

* Models: `M_{d,3}` for d in {3, 4, 5, 6, 8, 16, 32}, sigma in {0.002, 0.005, 0.01} (21 cells).
* Noise: `evaluate.noisy(base, sigma, seed, keyed=True)`, seeds 7000-7199 (200 models per cell).
* Both ladders on every model (`run_test_lean.sh`, tag `test7000`). Truth: `engine.exact_accuracy`,
  validated by plain Monte Carlo (2^19 samples per model, `mc_validate.py`) in all 21 cells.
* No exclusions. A model whose run crashes is rerun, not dropped. All 42 result files, the
  validation and the analysis are published whatever the outcome.

## Analysis (`analyze_test.py`, unchanged after this commit)

* Metric, per cell and ladder: the worst-case ratio, over all budgets B >= 1, of the ladder's MSE
  under the fair rule to sampling's MSE (`summarize.worst_ratios`, as in the README).
* Uncertainty: a paired percentile bootstrap over the 200 noisy models (the same resampled models
  for both ladders), 10,000 replicates, seed 0; 90% intervals.
* **Primary family**: for each of the 21 cells, H0 "the lean ladder's worst fair ratio is >= 1",
  p = bootstrap share of replicates >= 1, Holm at level 0.05 over the 21 cells. A cell counts as
  "lean beats sampling at every budget" only if its H0 is rejected.
* Secondary families (each Holm over 21 cells at 0.05): the same for `lguard` (a replication of the
  earlier test on new seeds and with the corrected accounting); and "lean has a smaller worst fair
  ratio than lguard", p = share of replicates with lean >= lguard.

## Expectations stated in advance (from the development seeds)

Likely to reject H0: `M_{6,3}` at all three noise levels, `M_{8,3}` at 0.002 and 0.005, `M_{4,3}`
at 0.002, `M_{5,3}` at 0.01, and the cells `lguard` already passed (`M_{5,3}`, `M_{16,3}`, `M_{32,3}`
at 0.002 and 0.005). Uncertain: `M_{8,3}` and `M_{16,3}` at 0.01, `M_{32,3}` at 0.01. Not expected:
`M_{4,3}` at 0.005 and 0.01, and `M_{3,3}` at every noise level. For any ladder whose first rung is
the prior "accuracy 1", the worst ratio is at least MSE(const) x cost(first rung) / var, which is
above 1 for those models.
