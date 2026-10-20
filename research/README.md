# Mechanistic accuracy estimates for AlgZoo RNNs that match random sampling

## The open problem

The [AlgZoo blog post](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters)
asks for **mechanistic estimates** of the accuracy of the zoo models: estimates that reason
deductively from the weights rather than from sampled inputs, and that are competitive with
random sampling in **mean squared error versus compute**. The evaluation protocol it suggests is
to add enough noise to the weights to change accuracy significantly, and to average the squared
error over the noisy models.

ARC reports mechanistic estimates for two 2nd-argmax models. The 10-parameter `M_{2,2}` is
fully explained and estimated with "a handful of computational operations". For the 32-parameter
`M_{4,3}`, its estimates "roughly match the computational efficiency of sampling". ARC describes
32 parameters as the largest model it more-or-less fully understands. The headline challenge,
the 432-parameter `M_{16,10}` (`example_2nd_argmax()`, a different checkpoint from
`zoo_2nd_argmax(16, 10)`), is still open (see [Limitations](#limitations)).

## Results

1. **Matching sampling on 3-input models up to 1,152 parameters, on held-out models.** For the
   1,152-parameter `M_{32,3}` (`zoo_2nd_argmax(32, 3)`), a fixed ladder of deterministic
   mechanistic estimators has **lower MSE than random sampling at every compute budget**, on 200
   fresh noisy models per noise level. The compute accounting, if anything, favours sampling. At
   σ = 0.002 and σ = 0.005 the worst-case MSE ratios to sampling are 0.13 and 0.37, with 90%
   bootstrap intervals far below 1. At σ = 0.01 the ratio is 0.86. Its interval is also below 1,
   but marginally: the one-sided bootstrap p ≈ 0.02 is not adjusted for the 21 cells tested, and
   a single extra model with about 2.2 times the largest error observed at the binding rung
   would push it above 1. The same holds, robustly, for the 320-parameter `M_{16,3}` at the two
   lower noise levels (0.25 and 0.72). It also holds for the 45-parameter `M_{5,3}` (0.29 and
   0.68), but `M_{5,3}` was not part of development, so that is a post-hoc observation from the
   test table. Six cells survive a Holm correction over all 21 at level 0.05 (their bootstrap
   p-values are below 1/2000): `M_{32,3}`, `M_{16,3}` and `M_{5,3}` at σ = 0.002 and 0.005. Note
   that ARC's `M_{4,3}` also has 3 inputs, and the cost of our method depends mainly on the
   number of inputs rather than on the number of parameters. From about 0.8–1.9 × 10³ forward
   passes, depending on the model, the estimate is exact (MSE ≈ 0). Argmedian with `n = 3` is
   the same task, and the zoo's argmedian `n = 3` models have identical weights.
2. **Exact accuracies, no sampling.** The same machinery computes the exact accuracy of all 37
   distinct AlgZoo RNN classifiers (2nd argmax and argmedian) with at most 5 inputs, up to the
   1,216-parameter `M_{32,5}` and `AM_{32,5}`. Run time grows with the number of linear regions:
   `M_{32,5}` has 24 million leaf regions, cut into 310 million simplicial cones, and took 36
   minutes on 9 processes (the root cones are independent). The answers are numerically
   converged. For every 4- and 5-input model, recomputing every Gaussian measure at higher order
   changes them by at most 4 × 10⁻¹¹. For 2- and 3-input models the measures are
   closed-form arcsines, accurate to about 10⁻⁹; cancellation on very thin cones costs a little,
   as an independent well-conditioned implementation showed. All 37 agree with 1.7 × 10⁷-sample
   Monte Carlo runs (|z| ≤ 2.4). The four largest 5-input models also agree root cone by root
   cone: with 6.7 × 10⁷ fresh samples, the χ² tests over their 240 root cones give p =
   0.07–0.92. Matching even 10⁻⁹ precision by sampling would take 10¹³–10¹⁷ samples, depending
   on the model's accuracy.
3. **A mechanistic explanation of the residual error, now including 5-input models: the
   order-truncated tie-sliver law.** The errors of well-trained models are thin wedges
   ("slivers") along the tie hyperplanes at which the target switches between two inputs. A
   first-order formula built from the logit gap on the tie hyperplane reproduces their error
   rate, once each sliver is cut off where it leaves its **order chamber** (the cone on which
   the ordering of the inputs, and so the target, is fixed). For all eight `n = 2` models,
   where it reduces to ARC's `M_{2,2}` explanation, it matches the exact error to within 1.3 ×
   10⁻⁶ relative (3 × 10⁻¹¹ absolute). It is within 3.4% for every zoo model with `n <= 4`, `d
   >= 6` and accuracy above 98% (13% without the cut-off). For the four 5-input zoo models with
   accuracy above 98% it is off by +4% to +19%, where the law without the cut-off was off by
   +18% to a factor of 5.6. Next to a triple tie a well-trained model can have a large logit
   gap on the tie hyperplane, and an uncut sliver then reaches far outside the chamber in which
   it is an error: for `M_{16,5}`, 0.1% of the points of the tie hyperplanes carried 68% of the
   uncut estimate. A Monte Carlo decomposition shows that the mechanism is the same at 5
   inputs. For those four models, 90–97% of the wrong inputs lie in exactly one sliver, 3–10%
   in two overlapping slivers (next to a triple tie) and at most 0.4% in none. What the law
   leaves out is second order in the sliver width: overlapping slivers are counted twice, and
   the logit gap bends inside a sliver. This residual is larger for the `d = 16` models, whose
   errors are 3–4 times larger than at `d = 32`. On all five training seeds of the larger 4-
   and 5-input models (35 models), the cap brings the law closer to the error every time, and
   at least 98% of the wrong inputs lie in one or two slivers. The law is less tight on some
   seeds, though: up to +25% for `M_{16,5}`, and +40% to +49% for the two least accurate seeds
   of `M_{8,4}` (97.2–97.6% accurate). The law costs 1–2% of the exact computation for 5-input
   models (1–2 × 10⁷ forward passes).

What is **not** solved: ARC's headline challenge `M_{16,10}`; matching sampling at intermediate
budgets for 4- and 5-input models, where exact computation costs 10⁵–10⁹ forward passes (the
sliver law does not help there: it describes trained models, and on noisy models it is off by
8–50%); among the 3-input models, the others (`M_{3,3}`, `M_{4,3}`, `M_{6,3}`, `M_{8,3}`) and
the heaviest noise level for `M_{5,3}` and `M_{16,3}`; and an exact second-order sliver law
(overlaps at triple ties, bending of the gap). See [Limitations](#limitations).

### 1. MSE versus compute (3-input models, held-out noisy-weights test)

![MSE vs compute for M_{32,3}](figures/mse_32_3.png)

![ratio to sampling for all 3-input models](figures/ratio_n3.png)

Worst-case ratio of our MSE to sampling's MSE over all budgets `B >= 1`, under the fair rule
(at each budget, the most expensive affordable rung of the fixed ladder; it never looks at the
truth). There are 200 held-out noisy models per row. The 90% interval is a percentile bootstrap
over noisy models. "Bootstrap share > 1" is the fraction of bootstrap replicates whose worst
ratio exceeds 1, a one-sided bootstrap p-value that is not adjusted for the 21 cells. **Bold**
means the whole interval is below 1. The binding rung is where the worst ratio occurs: the
ratio peaks just before that budget, when the next rung becomes affordable. The near-chance
`M_{2,3}` (accuracy 0.41) is not included.

<!-- SWEEP_TABLE -->
| model | params | σ | noisy models | mean accuracy | exact cost (fwd) | worst fair ratio | 90% CI | bootstrap share > 1 | binding rung (at budget) |
|---|---|---|---|---|---|---|---|---|---|
| `M_{3,3}` | 21 | 0.002 | 200 | 0.8458 | 790 | 15.54 | [14.92, 16.29] | 1.00 | roots4 (135) |
| `M_{3,3}` | 21 | 0.005 | 200 | 0.7979 | 783 | 20.99 | [19.78, 22.21] | 1.00 | roots4 (140) |
| `M_{3,3}` | 21 | 0.01 | 200 | 0.7191 | 767 | 34.27 | [31.57, 36.96] | 1.00 | roots6 (197) |
| `M_{4,3}` | 32 | 0.002 | 200 | 0.9776 | 862 | 1.78 | [1.73, 1.84] | 1.00 | roots4 (86) |
| `M_{4,3}` | 32 | 0.005 | 200 | 0.9623 | 846 | 2.97 | [2.82, 3.11] | 1.00 | roots4 (88) |
| `M_{4,3}` | 32 | 0.01 | 200 | 0.9311 | 843 | 6.43 | [5.98, 6.90] | 1.00 | roots8 (169) |
| `M_{5,3}` | 45 | 0.002 | 200 | 0.9949 | 841 | **0.29** | [0.28, 0.31] | 0.00 | roots6 (92) |
| `M_{5,3}` | 45 | 0.005 | 200 | 0.9874 | 827 | **0.68** | [0.64, 0.73] | 0.00 | roots6 (93) |
| `M_{5,3}` | 45 | 0.01 | 200 | 0.9741 | 814 | 1.32 | [1.24, 1.41] | 1.00 | roots6 (95) |
| `M_{6,3}` | 60 | 0.002 | 200 | 0.9977 | 799 | 6.70 | [5.60, 7.92] | 1.00 | any90 (725) |
| `M_{6,3}` | 60 | 0.005 | 200 | 0.9946 | 805 | 3.08 | [2.41, 3.83] | 1.00 | any90 (724) |
| `M_{6,3}` | 60 | 0.01 | 200 | 0.9892 | 805 | 1.78 | [1.38, 2.26] | 1.00 | any90 (723) |
| `M_{8,3}` | 96 | 0.002 | 200 | 0.9892 | 887 | 1.34 | [1.09, 1.61] | 0.99 | any128 (744) |
| `M_{8,3}` | 96 | 0.005 | 200 | 0.9717 | 895 | 1.21 | [1.15, 1.27] | 1.00 | roots6 (54) |
| `M_{8,3}` | 96 | 0.01 | 200 | 0.9416 | 894 | 3.14 | [2.92, 3.36] | 1.00 | roots8 (79) |
| `M_{16,3}` | 320 | 0.002 | 200 | 0.9880 | 1307 | **0.25** | [0.24, 0.27] | 0.00 | roots8 (41) |
| `M_{16,3}` | 320 | 0.005 | 200 | 0.9691 | 1295 | **0.72** | [0.68, 0.77] | 0.00 | roots8 (42) |
| `M_{16,3}` | 320 | 0.01 | 200 | 0.9347 | 1289 | 2.06 | [1.94, 2.18] | 1.00 | roots8 (45) |
| `M_{32,3}` | 1152 | 0.002 | 200 | 0.9940 | 1944 | **0.13** | [0.11, 0.15] | 0.00 | any512 (687) |
| `M_{32,3}` | 1152 | 0.005 | 200 | 0.9843 | 1923 | **0.37** | [0.33, 0.41] | 0.00 | any512 (690) |
| `M_{32,3}` | 1152 | 0.01 | 200 | 0.9693 | 1911 | **0.86** | [0.77, 0.97] | 0.02 | any512 (696) |
<!-- /SWEEP_TABLE -->

The ladder's weak point is its cheap rungs (`roots_k`, which linearise whole root cones) and,
for some of the other models, its mid-range rungs. A forward pass of `M_{3,3}`–`M_{8,3}` costs
only 70–350 flops, so the geometric overhead is relatively expensive. For `M_{3,3}` and
`M_{4,3}` the errors are not sliver-dominated (the tie-sliver law is off by −76% and −54%).
For `M_{6,3}` and `M_{8,3}`, whose errors are sliver-dominated, we do not yet understand why the
mid-range rungs fail.

### 2. Exact accuracies of ≤ 5-input models

`change at higher order` is how much the accuracy moves when every Gaussian measure is
recomputed at higher order (Plackett with 28 instead of 20 nodes per panel, cubature of degree 9
instead of 7), as a check on numerical precision. The cost is in forward-pass equivalents. The
Monte Carlo column uses 2²⁴ fresh samples; `z` is the difference in standard errors. For
`n <= 3` the measures are closed form, so the higher-order change is zero by construction.
The small models `M_{2,3}`, `M_{2,4}`, `M_{3,4}` and the 5-input models with `d <= 5` are far
from accurate and are included for completeness. `results/mc_roots.json` holds the per-root
Monte Carlo checks of the four largest models.

<!-- EXACT_TABLE -->
| model | params | exact accuracy | change at higher order | cost (fwd) | Monte Carlo (2^24) | z |
|---|---|---|---|---|---|---|
| `M_{2,2}` | 10 | 0.9999225685 | 0e+00 | 49.8 | 0.999921 | +0.77 |
| `M_{3,2}` | 18 | 0.9999735975 | 0e+00 | 45.2 | 0.999974 | -0.33 |
| `M_{4,2}` | 28 | 0.9999615632 | 0e+00 | 43.2 | 0.999961 | +0.55 |
| `M_{5,2}` | 40 | 0.9999768611 | 0e+00 | 42.1 | 0.999977 | -0.16 |
| `M_{6,2}` | 54 | 0.9999213211 | 0e+00 | 41.4 | 0.999920 | +0.65 |
| `M_{8,2}` | 88 | 0.9999777192 | 0e+00 | 40.7 | 0.999978 | -0.20 |
| `M_{16,2}` | 304 | 0.9999726792 | 0e+00 | 40 | 0.999973 | -0.02 |
| `M_{32,2}` | 1120 | 0.9999780121 | 0e+00 | 39.9 | 0.999978 | -0.31 |
| `M_{2,3}` | 12 | 0.4101712617 | 0e+00 | 328 | 0.410192 | -0.17 |
| `M_{3,3}` | 21 | 0.9083715809 | 0e+00 | 754 | 0.908343 | +0.40 |
| `M_{4,3}` | 32 | 0.9845503814 | 0e+00 | 865 | 0.984537 | +0.45 |
| `M_{5,3}` | 45 | 0.9991161910 | 0e+00 | 851 | 0.999108 | +1.09 |
| `M_{6,3}` | 60 | 0.9992213310 | 0e+00 | 774 | 0.999228 | -0.97 |
| `M_{8,3}` | 96 | 0.9994549030 | 0e+00 | 890 | 0.999454 | +0.08 |
| `M_{16,3}` | 320 | 0.9995081466 | 0e+00 | 1.34e+03 | 0.999504 | +0.70 |
| `M_{32,3}` | 1152 | 0.9995410620 | 0e+00 | 1.96e+03 | 0.999538 | +0.63 |
| `M_{2,4}` | 14 | 0.3280274101 | 6e-15 | 4.32e+05 | 0.328298 | -2.36 |
| `M_{3,4}` | 24 | 0.4332876450 | 8e-14 | 9.35e+05 | 0.433293 | -0.04 |
| `M_{4,4}` | 36 | 0.6020184118 | 1e-12 | 1.47e+06 | 0.601855 | +1.37 |
| `M_{5,4}` | 50 | 0.7456524969 | 7e-12 | 1.56e+06 | 0.745617 | +0.33 |
| `M_{6,4}` | 66 | 0.8647510736 | 5e-12 | 1.85e+06 | 0.864630 | +1.45 |
| `M_{8,4}` | 104 | 0.9874600108 | 1e-12 | 1.17e+06 | 0.987423 | +1.36 |
| `M_{16,4}` | 336 | 0.9983068360 | 6e-13 | 8.74e+05 | 0.998315 | -0.81 |
| `M_{32,4}` | 1184 | 0.9988002327 | 7e-13 | 7.46e+05 | 0.998792 | +0.97 |
| `M_{2,5}` | 16 | 0.3056348973 | 4e-14 | 1.61e+08 | 0.305603 | +0.28 |
| `M_{3,5}` | 27 | 0.3754419673 | 5e-12 | 1.55e+08 | 0.375440 | +0.01 |
| `M_{4,5}` | 40 | 0.4284504356 | 1e-11 | 2.06e+08 | 0.428510 | -0.49 |
| `M_{5,5}` | 55 | 0.5424323781 | 6e-12 | 4.29e+08 | 0.542502 | -0.57 |
| `M_{6,5}` | 72 | 0.7028572423 | 3e-11 | 7.44e+08 | 0.702825 | +0.29 |
| `M_{8,5}` | 112 | 0.8815144388 | 4e-11 | 7.09e+08 | 0.881561 | -0.59 |
| `M_{16,5}` | 352 | 0.9925277274 | 6e-12 | 8.59e+08 | 0.992489 | +1.83 |
| `M_{32,5}` | 1216 | 0.9972959155 | 3e-12 | 9.29e+08 | 0.997287 | +0.67 |
| `AM_{2,5}` | 16 | 0.2642094333 | 5e-14 | 2.99e+07 | 0.264125 | +0.78 |
| `AM_{4,5}` | 40 | 0.3939526176 | 7e-12 | 2.78e+08 | 0.393813 | +1.17 |
| `AM_{8,5}` | 112 | 0.6985511009 | 3e-11 | 5.45e+08 | 0.698462 | +0.80 |
| `AM_{16,5}` | 352 | 0.9831036499 | 1e-11 | 8.63e+08 | 0.983124 | -0.64 |
| `AM_{32,5}` | 1216 | 0.9955137455 | 4e-12 | 1.01e+09 | 0.995527 | -0.80 |
<!-- /EXACT_TABLE -->

### 3. The tie-sliver law

Tie-sliver estimates of the error rate (`sliver.py`) against the exact error rate of the trained
models: the first-order law, and the order-truncated law, whose slivers are cut off at the
boundary of their order chamber (see [Method](#method)). Both are computed exactly from the
cone geometry, and both agree with independent Monte Carlo estimates of the same formulas
(`test_sliver.py`, `sliver_mc.py law`; see [How this was checked](#how-this-was-checked)). The
last two columns are a Monte Carlo decomposition of the wrong inputs (`sliver_mc.py table`,
2²² samples). A wrong input lies in the sliver of
one of its target's walls (the tie hyperplanes bounding its order chamber where the target
changes) if the model is wrong all along the segment from the input to its foot on that wall.
The sum over walls of the true slivers is therefore `one + 2 × two`. The law differs from that
sum only through the bending of the logit gap inside the sliver, so
`law − exact = bending + two − none` (as fractions of the error). The last column is that
bending. For `M_{4,3}` and `M_{5,3}` the wrong inputs are still in slivers, but the slivers
are not thin, and the gap bends so much inside them that the first-order law fails. For
`M_{3,3}` and the weak 5-input models, 25–69% of the wrong inputs are where two slivers
overlap.

<!-- SLIVER_TABLE -->
| model | exact error | first-order law | relative difference | order-truncated law | relative difference | wrong inputs in one / two / no slivers | gap bending |
|---|---|---|---|---|---|---|---|
| `M_{2,2}` | 7.7431e-05 | 7.7431e-05 | +2.8e-08 | 7.7431e-05 | +2.8e-08 | - | - |
| `M_{3,2}` | 2.6402e-05 | 2.6402e-05 | +5.1e-08 | 2.6402e-05 | +5.1e-08 | - | - |
| `M_{4,2}` | 3.8437e-05 | 3.8437e-05 | +2.1e-08 | 3.8437e-05 | +2.1e-08 | - | - |
| `M_{5,2}` | 2.3139e-05 | 2.3139e-05 | +1.3e-06 | 2.3139e-05 | +1.3e-06 | - | - |
| `M_{6,2}` | 7.8679e-05 | 7.8679e-05 | +3.1e-08 | 7.8679e-05 | +3.1e-08 | - | - |
| `M_{8,2}` | 2.2281e-05 | 2.2281e-05 | +5.9e-08 | 2.2281e-05 | +5.9e-08 | - | - |
| `M_{16,2}` | 2.7321e-05 | 2.7321e-05 | +4.1e-08 | 2.7321e-05 | +4.1e-08 | - | - |
| `M_{32,2}` | 2.1988e-05 | 2.1988e-05 | +7.2e-08 | 2.1988e-05 | +7.2e-08 | - | - |
| `M_{3,3}` | 9.1628e-02 | 2.1552e-02 | -76.48% | 2.1552e-02 | -76.48% | 30.9% / 68.8% / 0.3% | -145.1% |
| `M_{4,3}` | 1.5450e-02 | 7.0544e-03 | -54.34% | 3.9999e-03 | -74.11% | 97.0% / 2.6% / 0.4% | -76.3% |
| `M_{5,3}` | 8.8381e-04 | 1.7219e-03 | +94.83% | 1.6882e-03 | +91.02% | 97.0% / 3.0% / 0.0% | +88.0% |
| `M_{6,3}` | 7.7867e-04 | 7.7931e-04 | +0.08% | 7.7896e-04 | +0.04% | 99.9% / 0.1% / 0.0% | -0.0% |
| `M_{8,3}` | 5.4510e-04 | 6.1569e-04 | +12.95% | 5.4459e-04 | -0.09% | 98.5% / 1.5% / 0.0% | -1.5% |
| `M_{16,3}` | 4.9185e-04 | 4.8772e-04 | -0.84% | 4.8713e-04 | -0.96% | 99.3% / 0.2% / 0.5% | -0.7% |
| `M_{32,3}` | 4.5894e-04 | 4.7612e-04 | +3.74% | 4.7432e-04 | +3.35% | 98.5% / 1.3% / 0.3% | +2.4% |
| `M_{6,4}` | 1.3525e-01 | 2.6462e-01 | +95.66% | 1.2740e-01 | -5.81% | 70.4% / 28.8% / 0.8% | -33.8% |
| `M_{8,4}` | 1.2540e-02 | 1.4127e-02 | +12.65% | 1.2440e-02 | -0.79% | 88.2% / 11.7% / 0.1% | -12.4% |
| `M_{16,4}` | 1.6932e-03 | 1.7793e-03 | +5.09% | 1.7131e-03 | +1.17% | 96.6% / 3.3% / 0.1% | -2.1% |
| `M_{32,4}` | 1.1998e-03 | 1.3373e-03 | +11.46% | 1.2038e-03 | +0.34% | 96.3% / 3.1% / 0.6% | -2.2% |
| `M_{6,5}` | 2.9714e-01 | 8.9346e-01 | +200.68% | 2.4752e-01 | -16.70% | 61.9% / 37.3% / 0.9% | -53.1% |
| `M_{8,5}` | 1.1849e-01 | 3.9084e-01 | +229.86% | 9.2101e-02 | -22.27% | 74.7% / 24.8% / 0.5% | -46.5% |
| `M_{16,5}` | 7.4723e-03 | 2.1148e-02 | +183.01% | 8.8514e-03 | +18.46% | 89.7% / 9.9% / 0.4% | +8.9% |
| `M_{32,5}` | 2.7041e-03 | 3.2140e-03 | +18.86% | 2.8476e-03 | +5.31% | 96.5% / 3.1% / 0.4% | +2.5% |
| `AM_{8,5}` | 3.0145e-01 | 4.9695e-01 | +64.85% | 1.5649e-01 | -48.09% | 50.7% / 48.8% / 0.5% | -96.3% |
| `AM_{16,5}` | 1.6896e-02 | 9.4125e-02 | +457.07% | 1.9704e-02 | +16.62% | 92.9% / 6.7% / 0.4% | +10.3% |
| `AM_{32,5}` | 4.4863e-03 | 5.2828e-03 | +17.76% | 4.6779e-03 | +4.27% | 95.1% / 4.7% / 0.2% | -0.2% |
<!-- /SLIVER_TABLE -->

**Other training seeds.** The zoo keeps the best of 5 training seeds. On all five seeds of the
larger 4- and 5-input models (`sliver_seeds.py`), the order-truncated law is closer to the
error than the first-order law on all 35 models. On every seed at least 98.4% of the wrong
inputs lie in one or two slivers. The law's accuracy tracks the model's. It is within 1.6% on
every seed of `M_{32,4}`, and within 9% on every seed of `M_{16,4}`, `M_{32,5}` and
`AM_{32,5}`. It is off by up to +17% for `AM_{16,5}`, +25% for `M_{16,5}`, and +40% to +49% for
the two least accurate seeds of `M_{8,4}` (accuracy 97.2–97.6%). For 4-input models the error
and the laws are exact. For 5-input models all three are Monte Carlo estimates (2²⁴ samples for
the error; 2²⁰ points per tie hyperplane for the laws; the `how` column gives the error's
relative standard error).

<!-- SEEDS_TABLE -->
| model | seed | error | how | first-order law | order-truncated law | wrong inputs in one / two / no slivers |
|---|---|---|---|---|---|---|
| `M_{8,4}` | 0 | 2.384e-02 | exact | +220.08% | +39.82% | 85.7% / 13.5% / 0.8% |
| `M_{8,4}` | 1 | 1.254e-02 | exact | +12.65% | -0.79% | 87.8% / 12.0% / 0.1% |
| `M_{8,4}` | 2 | 2.815e-02 | exact | +261.45% | +49.22% | 80.2% / 19.4% / 0.4% |
| `M_{8,4}` | 3 | 2.329e-02 | exact | +20.22% | +4.35% | 88.5% / 11.4% / 0.2% |
| `M_{8,4}` | 4 | 1.460e-02 | exact | +97.77% | +12.29% | 86.1% / 13.4% / 0.5% |
| `M_{16,4}` | 0 | 2.128e-03 | exact | +3.90% | +1.07% | 96.8% / 2.8% / 0.3% |
| `M_{16,4}` | 1 | 2.273e-03 | exact | +9.35% | +0.78% | 96.6% / 3.2% / 0.2% |
| `M_{16,4}` | 2 | 2.197e-03 | exact | +26.19% | +9.01% | 96.6% / 3.4% / 0.0% |
| `M_{16,4}` | 3 | 1.693e-03 | exact | +5.09% | +1.17% | 96.2% / 3.7% / 0.1% |
| `M_{16,4}` | 4 | 2.723e-03 | exact | +12.83% | +7.71% | 96.2% / 3.6% / 0.2% |
| `M_{32,4}` | 0 | 1.330e-03 | exact | +2.91% | -0.43% | 97.0% / 2.5% / 0.5% |
| `M_{32,4}` | 1 | 1.464e-03 | exact | +18.25% | +1.55% | 96.9% / 2.7% / 0.5% |
| `M_{32,4}` | 2 | 1.200e-03 | exact | +11.46% | +0.34% | 96.5% / 2.9% / 0.6% |
| `M_{32,4}` | 3 | 1.413e-03 | exact | +2.96% | -0.82% | 96.7% / 2.6% / 0.7% |
| `M_{32,4}` | 4 | 1.252e-03 | exact | +7.81% | +0.57% | 97.3% / 2.4% / 0.3% |
| `M_{16,5}` | 0 | 9.846e-03 | MC ±0.2% | +316.05% | -0.26% | 90.2% / 9.7% / 0.1% |
| `M_{16,5}` | 1 | 9.035e-03 | MC ±0.3% | +135.67% | +24.99% | 89.6% / 10.2% / 0.2% |
| `M_{16,5}` | 2 | 7.440e-03 | MC ±0.3% | +181.53% | +18.55% | 89.5% / 10.2% / 0.4% |
| `M_{16,5}` | 3 | 9.754e-03 | MC ±0.2% | +43.13% | +10.67% | 89.8% / 10.1% / 0.1% |
| `M_{16,5}` | 4 | 1.268e-02 | MC ±0.2% | +24.18% | +2.22% | 85.4% / 14.5% / 0.0% |
| `M_{32,5}` | 0 | 3.243e-03 | MC ±0.4% | +30.51% | +2.31% | 94.5% / 5.2% / 0.3% |
| `M_{32,5}` | 1 | 4.772e-03 | MC ±0.4% | +35.78% | +7.58% | 94.0% / 5.8% / 0.2% |
| `M_{32,5}` | 2 | 4.972e-03 | MC ±0.3% | +18.56% | +3.65% | 94.0% / 5.5% / 0.5% |
| `M_{32,5}` | 3 | 2.696e-03 | MC ±0.5% | +22.83% | +5.71% | 96.7% / 2.8% / 0.5% |
| `M_{32,5}` | 4 | 2.746e-03 | MC ±0.5% | +18.13% | +7.12% | 95.3% / 4.6% / 0.1% |
| `AM_{16,5}` | 0 | 1.688e-02 | MC ±0.2% | +459.24% | +16.73% | 92.9% / 6.7% / 0.3% |
| `AM_{16,5}` | 1 | 3.278e-02 | MC ±0.1% | +33.53% | +7.30% | 86.9% / 12.1% / 1.0% |
| `AM_{16,5}` | 2 | 3.190e-02 | MC ±0.1% | +37.01% | +6.27% | 86.8% / 12.5% / 0.7% |
| `AM_{16,5}` | 3 | 4.243e-02 | MC ±0.1% | +56.56% | +12.70% | 82.9% / 16.3% / 0.8% |
| `AM_{16,5}` | 4 | 4.687e-02 | MC ±0.1% | +77.74% | +5.10% | 81.9% / 16.5% / 1.6% |
| `AM_{32,5}` | 0 | 4.665e-03 | MC ±0.4% | +29.17% | +8.25% | 93.5% / 6.0% / 0.6% |
| `AM_{32,5}` | 1 | 5.855e-03 | MC ±0.3% | +18.32% | +0.10% | 94.2% / 5.4% / 0.5% |
| `AM_{32,5}` | 2 | 6.845e-03 | MC ±0.3% | +35.76% | +7.27% | 92.3% / 7.1% / 0.7% |
| `AM_{32,5}` | 3 | 4.490e-03 | MC ±0.4% | +20.96% | +4.10% | 95.1% / 4.7% / 0.2% |
| `AM_{32,5}` | 4 | 4.730e-03 | MC ±0.4% | +20.48% | +5.81% | 93.9% / 5.7% / 0.4% |
<!-- /SEEDS_TABLE -->

## Method

All AlgZoo RNNs are one-layer ReLU RNNs **without biases**, and the classification tasks
(2nd argmax, argmedian) depend only on the **ordering** of the inputs. Everything below follows
from those two facts; nothing samples inputs.

**Cones.** With no biases the network is positively homogeneous and piecewise linear, so
whether it is correct depends only on the direction of `x`, and each of its linear regions is
a polyhedral cone. Accuracy is the Gaussian measure (equivalently, the normalised solid angle)
of the union of cones on which the argmax of the logits is the target.

**Symmetry roots.** We start from the `2 n!` simplicial cones
`{x_{p1} >= ... >= x_{pn}, ±x_0 >= 0}`. On each, the target position is fixed (it is read off
the ordering `p`), and so is the first activation pattern (every step-0 pre-activation is
`w_i x_0`). The Gaussian measure of each is known in closed form. If `r` is the rank of `x_0`,
the `+x_0` cone has measure `P(at least r of n inputs are positive) / n!`, and the `−x_0` cone
has `1/n!` minus that.

**Exact refinement.** A cone whose extreme rays all agree on the sign of a pre-activation lies
on one side of that neuron's hyperplane: the pre-activation is linear on the cone, and every
point is a nonnegative combination of the rays. Processing time steps in order, the
pre-activations at step `t` are linear on any cone that has not been split before step `t`.
Where the rays disagree, we split the cone exactly along that hyperplane. Cones are kept as
extreme rays plus facets and split incrementally (the double description method), and children
reuse the parent's cached pre-activations so only new rays are evaluated. Each leaf is a cone on
which the logits are linear, and the region where the target loses is cut out by at most
`n - 1` more hyperplanes. The root cones are independent, so they are processed in parallel
(`exact_parallel.py`); the accuracy is the sum of their contributions.

**Gaussian measures.** Leaves are triangulated (pulling triangulation) into simplicial cones.
For `n <= 3` a simplicial cone's measure is a closed-form orthant probability. For `n >= 4`
there are two methods. Narrow cones (rays within 0.3 rad) use an identity: intersecting a cone
with the hyperplane `u·y = 1` gives

    gamma_n(K) = (1 / |S^{n-1}|) ∫_{flat simplex} |y|^{-n} dA,

the integral of a smooth function over a flat simplex. Degree-7 Grundmann–Möller cubature
computes it to high relative accuracy. Wide cones use Plackett's formula, a one-dimensional
integral with a closed-form integrand, on geometrically graded Gauss–Legendre panels and with
quantities computed from cofactors so they stay well conditioned. Plackett's formula loses
absolute accuracy (≈ 10⁻⁹) on very thin cones through cancellation, so any wide cone it
measures below 10⁻⁶ is recomputed by cubature after bisection. Accuracy is
`root measure − Σ (incorrect pieces)` per root, or `Σ (correct pieces)` for roots where the
model is wrong at the central ray. Regions where the model is wholly right or wholly wrong
therefore cost no measure computation.

**The tie-sliver mechanism.** In a well-trained model, almost all error lives in thin wedges
along the **tie hyperplanes** `x_p = x_q`, at which the target switches between positions `p`
and `q`. On the tie hyperplane the two logits should be equal. The model's actual gap there,
`Δ(x)`, divided by the rate `g` at which the gap grows moving off the hyperplane, is how far its
decision boundary is displaced: it is wrong for `0 < τ < −Δ(x)/g`. To first order this gives
the *tie-sliver law*

    error ≈ Σ_ties Σ_sides Σ_{pieces K of the model on the tie hyperplane}
                φ(0)/g_K · E[−Δ_K·ζ ; ζ ∈ K, Δ_K·ζ < 0],

where `ζ` is standard Gaussian on the tie hyperplane. By Gaussian integration by parts the
expectation reduces to measures of `(n − 2)`-dimensional facets (closed form for `n <= 5`). For
`n = 2` this is exactly ARC's `M_{2,2}` explanation. There, the "yellow" linear pieces, where
`a_0 ≈ a_1`, contain the tie line `x_0 = x_1`, and the error is the thin wedge between the
decision boundary and that line.

**The order cap.** A sliver is an error only inside its **order chamber**, the cone on which the
ordering of the inputs is fixed. Within a chamber the target is fixed, and the model's errors
are slivers on the target's two walls (the ties with the inputs ranked just above and just
below it). Moving a distance `τ` off the tie hyperplane `x_p = x_q`, one tied value rises and
the other falls by `τ/√2`. So the normal line leaves the chamber at
`ℓ(ζ) = min(√2 x_above − ζ_0, ζ_0 − √2 x_below)`, where `x_above` and `x_below` are the inputs
ranked just above and below the tie and `ζ_0 = √2 x_p`. On each root cone of the tie hyperplane
these neighbours are fixed, so `ℓ` is a minimum of linear functions of `ζ`. The
**order-truncated law** caps the width at the chamber boundary:

    error ≈ Σ_ties Σ_sides Σ_K φ(0) E[min(−Δ_K·ζ/g_K, ℓ(ζ)) ; ζ ∈ K, Δ_K·ζ < 0].

Splitting `K` into the cones on which each term of the minimum is the smallest reduces this to
expectations of linear functions over cones. In the implementation these (and the first-order
law's) are computed by triangulating each cone and integrating over flat slices of its
simplicial cones, as the exact engine does for thin cones. The facet formula above gives the
same values, but on very flat pieces its facet measures nearly cancel (see [How this was
checked](#how-this-was-checked)). For a well-trained model the cap is usually inactive, because
the slivers are far thinner than the chamber. Next to a triple tie, however, a model can have a
large gap on the tie hyperplane: for `M_{16,5}`, uncapped widths reach 30 where the true width
is 0.003. Those points dominate the uncapped law for 5-input models. The cap costs a few more
cone splits. What it leaves out is second order in the sliver width: slivers on a target's two
walls that overlap near a triple tie are counted twice, and the logit gap can bend inside a
sliver when a neuron switches.

**The anytime estimator (compute knob).** Exact refinement costs about 10³ forward-pass
equivalents for 3-input models. For smaller budgets there is a fixed ladder of cheaper rungs:

* `const` (cost 0) predicts accuracy 1, the prior that trained models are almost always right.
* `roots_k` analyses only the first `k` root cones, with no refinement, and leaves the rest at
  that prior. Unexamined cases are left at the prior rather than extrapolated from examined
  ones, in the spirit of ARC's
  ["Competing with sampling"](https://www.alignment.org/blog/competing-with-sampling/)
  discussion, where unchecked inputs get a default value instead of the average of checked ones.
* `any_k` refines cones exactly in order of decreasing angular width until `k` splits have been
  made, then finishes every remaining cone. A cone that is a single linear region is exact.
  Otherwise it is **linearised at its tie point**: each conflicting neuron takes the sign it has
  just inside the cone next to its tie face, where the cone's errors (tie slivers) live, or at
  the central ray if it has no tie face. As a guard against a linearisation that is badly wrong
  over a large cone, a cone with measure ≥ 0.01 whose tie linearisation claims more than 10% of
  it is wrong is also linearised at its central ray. If the two estimates differ by more than
  10% of the cone, the smaller one, which is closer to the prior, is used.
* `exact`.

The finishing rule was chosen on a development set of noisy models: seeds 32–95, which
include the failures found in review (see below), for `M_{16,3}` and `M_{32,3}` only. The
history, for transparency:

1. The plain tie rule (`tie`) and the always-on guard (`guard`) were tried first.
2. A guard that only fires when the tie estimate is implausible (`cguard`) and `tie` were then
   compared on the development set.
3. After seeing that comparison, `lguard` (the guard restricted to large cones) was designed to
   fix `cguard`'s failure on `M_{32,3}` at σ = 0.01.

The criterion was no catastrophic failures and all three `M_{32,3}` cells below 1. `tie` has
lower worst ratios on some cells, but fails `M_{32,3}` at σ = 0.005 because of one catastrophic
seed. `lguard` was then frozen and evaluated once on the fresh test set. Development-set worst
fair ratios (64 models per cell, seed-only noise):

<!-- DEV_TABLE -->
| model | σ | tie | cguard | lguard |
|---|---|---|---|---|
| `M_{32,3}` | 0.002 | 0.15 | 0.12 | 0.11 |
| `M_{32,3}` | 0.005 | 1.22 | 0.35 | 0.31 |
| `M_{32,3}` | 0.01 | 0.62 | 1.11 | 0.91 |
| `M_{16,3}` | 0.002 | 0.39 | 0.27 | 0.27 |
| `M_{16,3}` | 0.005 | 0.63 | 0.80 | 0.76 |
| `M_{16,3}` | 0.01 | 1.29 | 2.24 | 2.24 |
<!-- /DEV_TABLE -->

**Compute accounting.** Both sides are charged arithmetic operations: multiply-adds count as 2
flops, comparisons and reductions as 1, transcendental functions as 20. Sampling is charged one
*lean* forward pass per sample, `2d + (n − 1)(2d² + 2d) + 2·n_out·d`. This skips the product of
`W_hh` with the zero initial state, which our engine does not pay for either. Generating Gaussian
inputs and computing targets are not charged to sampling, which favours sampling. Our side is
charged for every necessary operation: pre-activations and their norms (cached per step and
shared by child cones), sign tests, splits, triangulation bookkeeping (charged at the cost
measured by tracing), cuts and Gaussian measures, root set-up, and the priority queue.

**Evaluation protocol.** ARC suggests perturbing the weights enough to change accuracy
significantly and averaging the squared error over the noisy models. We instantiate this by
adding `σ · RMS(tensor) · N(0, 1)` noise to each weight tensor, with σ ∈ {0.002, 0.005, 0.01}.
The noise is drawn from an RNG keyed by (seed, σ), so the three noise levels are independent
replications. The test set has **200 fresh noisy models per (model, σ)** (seeds 1000–1199),
untouched during development. Ground truth comes from the exact engine. The `exact` rung
therefore has zero MSE by construction, and the other rungs share code with it, so the engine
is validated independently against Monte Carlo: section 2 covers the trained models, and
`results/mc_validation.json` holds 2¹⁹ fresh samples for each of the 1,800 held-out models in
the `M_{5,3}`, `M_{16,3}` and `M_{32,3}` cells. Across the nine cells the χ² p-values are
0.05–1.00. The largest |z| among the 1,800 models (3.84) shrinks to −1.46 when that model is
re-checked with 2²⁴ samples. Sampling with `B` forward passes has MSE
`mean_k p_k (1 − p_k) / B`.

For a budget `B` the **fair rule** runs the most expensive rung of the fixed ladder that costs at
most `B` on average; it never looks at the truth. Between rung costs the ratio of its MSE to
sampling's grows with `B`, so we compute the worst ratio exactly, just below each next rung's
cost, over all `B >= 1`. We also give a 90% bootstrap interval (resampling noisy models) and the
bootstrap probability that the worst ratio exceeds 1.

## How this was checked

The first version of this write-up claimed that `M_{16,3}` and `M_{32,3}` beat sampling at
every budget, at all three noise levels, on 32 noisy models each. An adversarial review found
three real problems. It used five independent AI review agents (Claude subagents), each with a
different lens, and two skeptic agents tried to refute every finding:

1. **Asymmetric accounting.** Sampling was charged for `W_hh · h_0` with `h_0 = 0`, which the
   engine never paid for. Fixing this raised every ratio by about 1.4×, and on those 32 models
   `M_{16,3}` no longer passed at σ = 0.002 and σ = 0.01.
2. **A rare catastrophic failure.** On fresh noise seeds, one `M_{32,3}` model had a whole
   root cone mis-linearised (estimate 0.888 against a true 0.988). On those seeds (32–63) the
   worst ratio at σ = 0.005 rose to 1.66; it is 1.22 over seeds 32–95 with the final accounting
   (development table, `tie`). The large-cone guard above was added to fix this.
3. **No uncertainty.** 32 models with heavy-tailed errors cannot support an "every budget"
   claim, and the three noise levels had reused the same noise directions.

The review found no leakage of ground truth into the estimates or the rung choice. In a check
whose outputs were not saved, ground truth matched Monte Carlo on all 192 models of the
original headline settings; `results/mc_validation.json` now saves the corresponding check for
the held-out models. A second review of the final version found no further substantive
problems. Its corrections are reflected here: multiplicity, the history of the rule choice,
`n <= 3` precision, and wording. Everything above comes from the corrected accounting, the
frozen rule and the fresh test set.

**The order cap of the sliver law.** The 5-input failure of the first-order law was first
checked to be real: an independent Monte Carlo estimate of the same formula reproduces it (2.07
× 10⁻² against 2.11 × 10⁻² for `M_{16,5}`). Monte Carlo then showed that the errors are slivers
between adjacent ranks: over 99% of the wrong predictions of the four accurate 5-input models
are the input ranked just above or just below the target, and the median gap between the two
values is 0.001–0.007. The uncut estimate is dominated by points where the true sliver width is
a thousandth of the first-order width or less. The cap has no free parameter. Three variants
were compared on the zoo models by Monte Carlo before the exact implementation:

* capping where the normal line leaves the order chamber (the law reported here);
* capping only where the target itself changes, which differs at second order (+20% instead
  of +18% for `M_{16,5}`);
* also capping where another class overtakes the competitor, so that overlapping slivers are
  counted once, which undercounts (−14% for `M_{16,5}`, −28% for `AM_{32,5}`).

The chamber cap matches the geometry of the decomposition and was the most accurate of the
three, so this is a choice made after seeing the 5-input results. It was then checked on the
other training seeds (seeds 0–4), which were not used to choose it. The exact implementation
agrees with its Monte Carlo estimate on every model tested: |z| ≤ 1.7 for the `n <= 4` models
in `test_sliver.py`, and |z| ≤ 2.1 for every tie hyperplane of `M_{16,5}` and `M_{32,5}`. (For
5-input models the Monte Carlo standard errors are optimistic, because the widths are
heavy-tailed: two independent runs for `M_{16,5}` differ by 2σ.) That per-tie comparison found
one flaw in the first exact implementation, now fixed. It computed each piece's contribution as
`w · E[ζ 1_K]`, with the first moment from facet measures. On a few flat pieces (wedges about
10⁻⁸ thick next to the zero set of a large gap with a tiny growth rate) the facet measures
nearly cancel, and the width functional, of norm about 10⁶, amplified their rounding error into
spurious contributions of about 10⁻⁵: 2% of `M_{32,5}`'s tie `x_0 = x_4` (z = +12.5). Each
piece is now triangulated and integrated by slice cubature (`slice.slice_linear_moment`). This
changed `M_{32,5}`'s truncated law from +5.6% to +5.3%, and no other value at the precision
shown.

## Limitations

* **`M_{16,10}` is still open.** The exact engine refines every linear region, and their number
  grows exponentially with sequence length. Sampling 200k inputs of `M_{16,10}` gives 138k
  distinct activation-pattern sequences, so exact refinement is hopeless there. Its error is
  also not tie-sliver dominated: about a fifth comes from inputs with at most one positive
  value, and a substantial part from large gaps. A cheaper understanding of the dense neurons
  is still needed.
* **Intermediate budgets for `n >= 4`.** Exact computation of 4- and 5-input models costs
  10⁵–10⁹ forward-pass equivalents. Above that budget it beats sampling by many orders of
  magnitude. Below it our approximate rungs were not competitive: in exploratory runs on
  `M_{16,4}`, not included in `results/`, they were 10–500× worse than sampling in MSE. In 4-D
  the unresolved cones are large, and linearising them is unreliable.
* **Small 3-input models and the heaviest noise.** The fair ladder loses to sampling over a
  window of budgets for `M_{3,3}`, `M_{4,3}`, `M_{6,3}` and `M_{8,3}` at every noise level, and
  for `M_{5,3}` and `M_{16,3}` at σ = 0.01 (see the table). It always wins once exact
  computation is affordable, at about 10³ forward passes. For these models a forward pass is
  cheap (70–1,200 flops), so the geometric overhead of the cheap rungs is relatively expensive.
  For `M_{3,3}` and `M_{4,3}` the errors are not sliver-dominated. For `M_{6,3}` and `M_{8,3}`
  we do not yet know why the mid-range rungs fail. ARC's own estimates for `M_{4,3}` "roughly
  match the computational efficiency of sampling"; ours do not at intermediate budgets.
* **The sliver law explains trained models, not noisy ones.** It describes the errors of the
  zoo models as trained. Under the weight noise of the evaluation protocol the 4-input errors
  grow 6–40 times (`M_{16,4}`: from 0.17% to 1–7% at σ = 0.002–0.01), and on 9 such noisy
  models the order-truncated law is off by +8% to +50%. So it is not an estimator that competes
  with sampling under that protocol. Its remaining error on trained models is second order and
  is measured above by Monte Carlo, not computed. An exact second-order law (overlapping slivers
  at triple ties, bending of the gap) is still to be written.
* **The estimator rule was chosen on a development set.** The finishing rule was chosen among
  three candidates on development seeds 32–95, then frozen; all reported numbers come from
  fresh seeds 1000–1199. The noise model, its scale and the number of models are our own
  instantiation of ARC's qualitative suggestion. The ladder builds in the prior that trained
  models are nearly perfect (the `const` and `roots_k` rungs). Against a sampler given the same
  prior through shrinkage, the margins at the cheapest budgets would be smaller.
* **Compute is counted, not timed.** Both sides are charged arithmetic operations. Our
  Python implementation is not fast in wall-clock terms.

## Reproducing

```bash
pip install -e .            # from the repo root; plus numpy, scipy, matplotlib
cd research
python test_engine.py M2,2 M4,3 M8,4          # exact engine vs Monte Carlo
python test_sliver.py M:8,3 M:16,4               # sliver laws vs their Monte Carlo estimates
./run_tables.sh                                 # exact accuracies (n <= 5), tie-sliver laws, per-root MC
python sliver_mc.py table                        # where the wrong inputs lie (one / two / no slivers)
python sliver_seeds.py                           # sliver laws on training seeds 0-4
./run_dev.sh                                     # development set used to choose the finishing rule
./run_test.sh                                    # held-out test: 7 models x 3 noise levels x 200 models
python summarize.py results/test/*.jsonl         # worst-case ratios with bootstrap intervals
python mc_validate.py results/test/M_{5,16,32}_3_*.jsonl  # independent Monte Carlo check of ground truth
python plot.py 32 16                             # figures
python make_tables.py --readme                   # refill the tables in this README
```

All runs are resumable (results are appended per model, and per root cone in
`results/roots/`). Files: `engine.py` (cones, refinement, exact estimator, anytime estimator
and ladder), `exact_parallel.py` (the exact estimator over root cones in a process pool),
`orthant.py` (orthant probabilities), `slice.py` (slice cubature), `sliver.py` (tie-sliver
laws), `sliver_mc.py` (Monte Carlo checks of the sliver laws and the decomposition of the wrong
inputs), `sliver_seeds.py` (the laws on other training seeds), `evaluate.py` (noisy-ensemble protocol), `summarize.py`, `plot.py`, `make_tables.py`,
`exact_table.py`, `sliver_table.py`, `mc_validate.py`, `mc_roots.py` (per-root Monte Carlo
check), `test_engine.py`, `test_sliver.py`.
