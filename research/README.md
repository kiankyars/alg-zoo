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
3. **A mechanistic account of where the trained models' errors are: the exact tie-sliver
   law.** The errors of well-trained models are thin wedges ("slivers") along the tie
   hyperplanes at which the target switches between two inputs. Inside an **order chamber**
   (the cone on which the ordering of the inputs, and so the target, is fixed) the target has
   two such walls; give each wrong input to the nearer one. From every point of a wall, follow
   the normal line through the model's linear regions and measure exactly the stretch on which
   the model is wrong. The resulting **exact sliver law** is the Gaussian measure of a set of
   wrong inputs, computed from the weights by cone geometry alone. So it can only fall short of
   the error, and the shortfall is exactly the wrong inputs that are not reached from their
   wall. For the eight 2-input models, where it reduces to ARC's `M_{2,2}` explanation, it
   equals the exact error to within 1.3 × 10⁻⁶ relative, the precision of the closed-form
   arcsine reference. For every 3-input zoo model with `d >= 5` and every 4-input one with
   `d >= 6` it accounts for 99.1–100% of the error, except the 86%-accurate `M_{6,4}` (97.2%);
   it was not computed for the 75%-accurate `M_{5,4}`. It accounts for 98.4% of ARC's `M_{4,3}`
   and 98.8% of the 91%-accurate `M_{3,3}`, whose errors the order-truncated law misses by 74%
   and 76%. For the four
   5-input zoo models with accuracy above 98%, a Monte Carlo estimate of it accounts for
   98.8–99.6% of the error; computing it exactly there would take hundreds of CPU-hours.

   The law also explains why simpler versions fail. The *first-order* law extrapolates the logit
   gap linearly off the tie hyperplane; the *order-truncated* law also cuts each sliver where it
   leaves its chamber. That cut matters: next to a triple tie a well-trained model can have a
   large gap on the tie hyperplane, and for `M_{16,5}` 0.1% of the points of the tie
   hyperplanes carried 68% of the uncut estimate. The order-truncated law is within 3.4% for
   every zoo model with `n <= 4`, `d >= 6` and accuracy above 98%, and costs 2–10% of the exact
   computation for 4-input models, but it is off by +4% to +18% for the accurate 5-input
   models. Its error splits exactly into three second-order parts: **overlaps** (near a triple
   tie it counts the slivers of the target's two walls twice), **bending** (the gap is not
   linear inside a sliver once a neuron switches) and **unattached** errors (missed by the exact
   law). For the `n <= 4` models with `d >= 6` and accuracy above 98%, each part is at most 4%
   of the error; for `M_{4,3}` and `M_{5,3}` the bending is −74% and +89%, because neurons
   switch inside their slivers, so the gap is far from linear there. For the accurate 5-input
   models the order-truncated law's excess
   splits into overlaps (+2% to +15% of the error) and bending (+3% to +11%), with unattached
   errors of 0.4–1.2%. On all five training seeds of the larger 4- and 5-input models (35
   models), the exact sliver law is within 2.4% of the error, where the order-truncated law is
   off by up to +49%.

What is **not** solved: ARC's headline challenge `M_{16,10}`; matching sampling at intermediate
budgets for 4- and 5-input models, where exact computation costs 10⁵–10⁹ forward passes (the
sliver laws do not help there: they describe trained models, the order-truncated law is off by
+8% to +50% on 9 noisy 4-input models, and the exact sliver law costs 0.3–24 times as much as
exact computation for 4-input models and far more for 5-input models); and,
among the 3-input models, the others (`M_{3,3}`, `M_{4,3}`, `M_{6,3}`, `M_{8,3}`) and the
heaviest noise level for `M_{5,3}` and `M_{16,3}`. See [Limitations](#limitations).

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

### 3. The tie-sliver laws

Tie-sliver estimates of the error rate of the trained models (`sliver.py`), as relative
differences from the exact error rate. All four laws are computed from the weights by cone
geometry (see [Method](#method)), except the exact sliver law of the 5-input models, which is
a Monte Carlo estimate (see below):

* **first-order law**: the width of the sliver at a point of the tie hyperplane is the logit
  gap there divided by the rate at which the gap grows off the hyperplane;
* **order-truncated law**: the same width, cut off where the normal line leaves the order
  chamber;
* **bisector law**: the same width, cut off at the bisector between the target's two walls (a
  third of the way to the other wall) and at the chamber's other exits, so that the slivers of
  different walls are disjoint;
* **exact sliver law**: the bisector cap, with the true length of the wrong stretch along each
  normal line instead of the linear extrapolation.

The next three columns split the order-truncated law's error exactly,
`truncated − exact = overlaps + bending − unattached`, with overlaps = truncated − bisector
law, bending = bisector law − exact sliver law and unattached = exact error − exact sliver law
(all as fractions of the exact error). For wide slivers, bending also includes the decay of the
Gaussian density across the sliver, which the first-order laws ignore. Unattached errors cannot
be negative, because the exact sliver law measures a set of wrong inputs; the tiny negative values for `n = 2` are the
reference's precision. The last column is an independent Monte Carlo view of the same
geometry (`sliver_mc.py table`, 2²² samples). A wrong input lies in the sliver of one of its
target's walls if the model is wrong all along the segment from the input to its foot on that
wall; "two" means it lies in the slivers of both walls, next to a triple tie. For the 5-input
models the exact sliver law is a Monte Carlo estimate (marked MC, with its standard error
relative to the error): true widths found by a grid and bisection on 2¹⁷ normal lines per tie
hyperplane (`sliver_mc.py widths-table`), which agrees point by point with the exact walk on
the 3- and 4-input models. Computing it exactly would take hundreds of CPU-hours per model: a
single (root cone, side) unit of one tie of `M_{16,5}`, out of 120 per tie (60 root cones, two
sides each), ran for more than 30 CPU-minutes.

For `M_{4,3}` and `M_{5,3}` the first-order laws fail because the gap bends inside the slivers
(bending −74% and +89%), but the errors are still slivers: the exact sliver law accounts for
98.4% and 99.8% of them. For `M_{3,3}` and the weak 5-input models, 25–69% of the wrong inputs
are where two slivers overlap.

<!-- SLIVER_TABLE -->
| model | exact error | first-order law | order-truncated law | bisector law | exact sliver law | overlaps | bending | unattached | in one / two / no slivers |
|---|---|---|---|---|---|---|---|---|---|
| `M_{2,2}` | 7.7431e-05 | +2.8e-08 | +2.8e-08 | +2.8e-08 | +1.8e-09 | +0.0e+00 | +2.6e-08 | -1.8e-09 | - |
| `M_{3,2}` | 2.6402e-05 | +5.1e-08 | +5.1e-08 | +5.1e-08 | +4.4e-08 | +0.0e+00 | +7.2e-09 | -4.4e-08 | - |
| `M_{4,2}` | 3.8437e-05 | +2.1e-08 | +2.1e-08 | +2.1e-08 | +1.5e-08 | +0.0e+00 | +5.8e-09 | -1.5e-08 | - |
| `M_{5,2}` | 2.3139e-05 | +1.3e-06 | +1.3e-06 | +1.3e-06 | +1.3e-06 | +0.0e+00 | +6.9e-09 | -1.3e-06 | - |
| `M_{6,2}` | 7.8679e-05 | +3.1e-08 | +3.1e-08 | +3.1e-08 | +3.2e-09 | +0.0e+00 | +2.7e-08 | -3.2e-09 | - |
| `M_{8,2}` | 2.2281e-05 | +5.9e-08 | +5.9e-08 | +5.9e-08 | +5.5e-08 | +0.0e+00 | +4.0e-09 | -5.5e-08 | - |
| `M_{16,2}` | 2.7321e-05 | +4.1e-08 | +4.1e-08 | +4.1e-08 | +3.4e-08 | +0.0e+00 | +6.4e-09 | -3.4e-08 | - |
| `M_{32,2}` | 2.1988e-05 | +7.2e-08 | +7.2e-08 | +7.2e-08 | +6.8e-08 | +0.0e+00 | +4.1e-09 | -6.8e-08 | - |
| `M_{3,3}` | 9.1628e-02 | -76.48% | -76.48% | -77.07% | -1.20% | +0.59% | -75.87% | +1.20% | 30.9% / 68.8% / 0.3% |
| `M_{4,3}` | 1.5450e-02 | -54.34% | -74.11% | -75.54% | -1.56% | +1.43% | -73.98% | +1.56% | 97.0% / 2.6% / 0.4% |
| `M_{5,3}` | 8.8381e-04 | +94.83% | +91.02% | +89.22% | -0.22% | +1.80% | +89.44% | +0.22% | 97.0% / 3.0% / 0.0% |
| `M_{6,3}` | 7.7867e-04 | +0.08% | +0.04% | +0.02% | -1.1e-05 | +0.02% | +0.02% | +1.1e-05 | 99.9% / 0.1% / 0.0% |
| `M_{8,3}` | 5.4510e-04 | +12.95% | -0.09% | -1.40% | -0.14% | +1.31% | -1.26% | +0.14% | 98.5% / 1.5% / 0.0% |
| `M_{16,3}` | 4.9185e-04 | -0.84% | -0.96% | -1.05% | -0.51% | +0.09% | -0.54% | +0.51% | 99.3% / 0.2% / 0.5% |
| `M_{32,3}` | 4.5894e-04 | +3.74% | +3.35% | +3.13% | -0.76% | +0.22% | +3.90% | +0.76% | 98.5% / 1.3% / 0.3% |
| `M_{6,4}` | 1.3525e-01 | +95.66% | -5.81% | -22.18% | -2.82% | +16.38% | -19.36% | +2.82% | 70.4% / 28.8% / 0.8% |
| `M_{8,4}` | 1.2540e-02 | +12.65% | -0.79% | -4.03% | -0.38% | +3.24% | -3.65% | +0.38% | 88.2% / 11.7% / 0.1% |
| `M_{16,4}` | 1.6932e-03 | +5.09% | +1.17% | -0.17% | -0.63% | +1.34% | +0.46% | +0.63% | 96.6% / 3.3% / 0.1% |
| `M_{32,4}` | 1.1998e-03 | +11.46% | +0.34% | -1.32% | -0.90% | +1.66% | -0.43% | +0.90% | 96.3% / 3.1% / 0.6% |
| `M_{6,5}` | 2.9714e-01 | +200.68% | -16.70% | -42.96% | -6.01% (MC ±0.2%) | +26.26% | -36.95% | +6.01% | 61.9% / 37.3% / 0.9% |
| `M_{8,5}` | 1.1849e-01 | +229.86% | -22.27% | -33.39% | -5.01% (MC ±0.2%) | +11.12% | -28.38% | +5.01% | 74.7% / 24.8% / 0.5% |
| `M_{16,5}` | 7.4723e-03 | +183.01% | +18.46% | +3.87% | -1.23% (MC ±0.3%) | +14.58% | +5.11% | +1.23% | 89.7% / 9.9% / 0.4% |
| `M_{32,5}` | 2.7041e-03 | +18.86% | +5.31% | +2.90% | -0.40% (MC ±0.2%) | +2.41% | +3.30% | +0.40% | 96.5% / 3.1% / 0.4% |
| `AM_{8,5}` | 3.0145e-01 | +64.85% | -48.09% | -60.03% | -6.21% (MC ±0.1%) | +11.94% | -53.82% | +6.21% | 50.7% / 48.8% / 0.5% |
| `AM_{16,5}` | 1.6896e-02 | +457.07% | +16.62% | +9.99% | -0.71% (MC ±0.1%) | +6.63% | +10.70% | +0.71% | 92.9% / 6.7% / 0.4% |
| `AM_{32,5}` | 4.4863e-03 | +17.76% | +4.27% | +2.16% | -0.36% (MC ±0.2%) | +2.11% | +2.52% | +0.36% | 95.1% / 4.7% / 0.2% |
<!-- /SLIVER_TABLE -->

**Other training seeds.** The zoo keeps the best of 5 training seeds. On all five seeds of the
larger 4- and 5-input models (35 models, `sliver_seeds.py`), the exact sliver law is within
2.4% of the error. It is −0.2% to −2.3% for the 4-input models, where both are exact, and
−2.2% to +0.5% for the 5-input models, where both are Monte Carlo estimates (the one positive
value is within its standard error of 0.6%). The order-truncated law is off by up to +25% for
`M_{16,5}` and by +40% and +49% for the two least accurate seeds of `M_{8,4}` (accuracy
97.2–97.6%). There most of the excess is overlapping slivers (24 and 38 points of the error; 14%
and 19% of their wrong inputs lie in two slivers), and the rest is bending. On every seed at
least 98.4% of the wrong inputs lie in one or two slivers. For 5-input models the error is
estimated from 2²⁴ samples (the `how` column gives its relative standard error), the
first-order laws from 2²⁰ points per tie hyperplane and the exact sliver law from 2¹⁵ normal
lines per tie hyperplane.

<!-- SEEDS_TABLE -->
| model | seed | error | how | first-order law | order-truncated law | bisector law | exact sliver law | in one / two / no slivers |
|---|---|---|---|---|---|---|---|---|
| `M_{8,4}` | 0 | 2.384e-02 | exact | +220.08% | +39.82% | +16.01% | -2.14% | 85.7% / 13.5% / 0.8% |
| `M_{8,4}` | 1 | 1.254e-02 | exact | +12.65% | -0.79% | -4.03% | -0.38% | 87.8% / 12.0% / 0.1% |
| `M_{8,4}` | 2 | 2.815e-02 | exact | +261.45% | +49.22% | +11.56% | -2.33% | 80.2% / 19.4% / 0.4% |
| `M_{8,4}` | 3 | 2.329e-02 | exact | +20.22% | +4.35% | -1.27% | -0.75% | 88.5% / 11.4% / 0.2% |
| `M_{8,4}` | 4 | 1.460e-02 | exact | +97.77% | +12.29% | -1.05% | -1.15% | 86.1% / 13.4% / 0.5% |
| `M_{16,4}` | 0 | 2.128e-03 | exact | +3.90% | +1.07% | -0.04% | -0.57% | 96.8% / 2.8% / 0.3% |
| `M_{16,4}` | 1 | 2.273e-03 | exact | +9.35% | +0.78% | -1.51% | -0.33% | 96.6% / 3.2% / 0.2% |
| `M_{16,4}` | 2 | 2.197e-03 | exact | +26.19% | +9.01% | +5.58% | -0.18% | 96.6% / 3.4% / 0.0% |
| `M_{16,4}` | 3 | 1.693e-03 | exact | +5.09% | +1.17% | -0.17% | -0.63% | 96.2% / 3.7% / 0.1% |
| `M_{16,4}` | 4 | 2.723e-03 | exact | +12.83% | +7.71% | +6.39% | -0.48% | 96.2% / 3.6% / 0.2% |
| `M_{32,4}` | 0 | 1.330e-03 | exact | +2.91% | -0.43% | -1.28% | -0.63% | 97.0% / 2.5% / 0.5% |
| `M_{32,4}` | 1 | 1.464e-03 | exact | +18.25% | +1.55% | +0.24% | -0.77% | 96.9% / 2.7% / 0.5% |
| `M_{32,4}` | 2 | 1.200e-03 | exact | +11.46% | +0.34% | -1.32% | -0.90% | 96.5% / 2.9% / 0.6% |
| `M_{32,4}` | 3 | 1.413e-03 | exact | +2.96% | -0.82% | -1.71% | -0.86% | 96.7% / 2.6% / 0.7% |
| `M_{32,4}` | 4 | 1.252e-03 | exact | +7.81% | +0.57% | -1.03% | -0.56% | 97.3% / 2.4% / 0.3% |
| `M_{16,5}` | 0 | 9.846e-03 | MC ±0.2% | +316.05% | -0.26% | -3.97% | -0.26% | 90.2% / 9.7% / 0.1% |
| `M_{16,5}` | 1 | 9.035e-03 | MC ±0.3% | +135.67% | +24.99% | +6.44% | -0.45% | 89.6% / 10.2% / 0.2% |
| `M_{16,5}` | 2 | 7.440e-03 | MC ±0.3% | +181.53% | +18.55% | +4.36% | -1.54% | 89.5% / 10.2% / 0.4% |
| `M_{16,5}` | 3 | 9.754e-03 | MC ±0.2% | +43.13% | +10.67% | +3.84% | -1.11% | 89.8% / 10.1% / 0.1% |
| `M_{16,5}` | 4 | 1.268e-02 | MC ±0.2% | +24.18% | +2.22% | -4.79% | -0.24% | 85.4% / 14.5% / 0.0% |
| `M_{32,5}` | 0 | 3.243e-03 | MC ±0.4% | +30.51% | +2.31% | -0.33% | -1.41% | 94.5% / 5.2% / 0.3% |
| `M_{32,5}` | 1 | 4.772e-03 | MC ±0.4% | +35.78% | +7.58% | +3.63% | -1.03% | 94.0% / 5.8% / 0.2% |
| `M_{32,5}` | 2 | 4.972e-03 | MC ±0.3% | +18.56% | +3.65% | +0.91% | -0.96% | 94.0% / 5.5% / 0.5% |
| `M_{32,5}` | 3 | 2.696e-03 | MC ±0.5% | +22.83% | +5.71% | +3.13% | -0.46% | 96.7% / 2.8% / 0.5% |
| `M_{32,5}` | 4 | 2.746e-03 | MC ±0.5% | +18.13% | +7.12% | +4.54% | +0.53% | 95.3% / 4.6% / 0.1% |
| `AM_{16,5}` | 0 | 1.688e-02 | MC ±0.2% | +459.24% | +16.73% | +10.20% | -0.60% | 92.9% / 6.7% / 0.3% |
| `AM_{16,5}` | 1 | 3.278e-02 | MC ±0.1% | +33.53% | +7.30% | -1.23% | -1.80% | 86.9% / 12.1% / 1.0% |
| `AM_{16,5}` | 2 | 3.190e-02 | MC ±0.1% | +37.01% | +6.27% | -2.05% | -1.90% | 86.8% / 12.5% / 0.7% |
| `AM_{16,5}` | 3 | 4.243e-02 | MC ±0.1% | +56.56% | +12.70% | -0.54% | -1.42% | 82.9% / 16.3% / 0.8% |
| `AM_{16,5}` | 4 | 4.687e-02 | MC ±0.1% | +77.74% | +5.10% | -5.82% | -2.18% | 81.9% / 16.5% / 1.6% |
| `AM_{32,5}` | 0 | 4.665e-03 | MC ±0.4% | +29.17% | +8.25% | +2.32% | -1.14% | 93.5% / 6.0% / 0.6% |
| `AM_{32,5}` | 1 | 5.855e-03 | MC ±0.3% | +18.32% | +0.10% | -2.30% | -0.82% | 94.2% / 5.4% / 0.5% |
| `AM_{32,5}` | 2 | 6.845e-03 | MC ±0.3% | +35.76% | +7.27% | +0.84% | -1.04% | 92.3% / 7.1% / 0.7% |
| `AM_{32,5}` | 3 | 4.490e-03 | MC ±0.4% | +20.96% | +4.10% | +2.04% | -0.85% | 95.1% / 4.7% / 0.2% |
| `AM_{32,5}` | 4 | 4.730e-03 | MC ±0.4% | +20.48% | +5.81% | +2.77% | -0.62% | 93.9% / 5.7% / 0.4% |
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

**The exact sliver law.** Both second-order effects can be computed exactly. *Overlaps*: give
each wrong input to the nearer of its target's two walls. The walls (the target's ties with
the inputs ranked just above and just below it) meet at 60°, since their normals
`(e_p − e_q)/√2` and `(e_q − e_r)/√2` have inner product −1/2. So the normal line from one wall
crosses the bisector a third of the way to the other wall. The **bisector cap** is
`min(ℓ_target / 3, ℓ_other)`, where `ℓ_target` is the order limit at which the line reaches
the target's other wall and `ℓ_other` the chamber's other exit. With it the slivers of
different walls, sides and chambers are disjoint. *Bending*: replace the linear width
`−Δ/g` by the true one. Along the normal line `ζ + τν` from a piece `Q` of the tie hyperplane,
every logit gap and pre-activation is linear in `(ζ, τ)` within the linear region the line is
in. So every neuron flip, every zero of a logit gap and the caps are each at a `τ` that is a
linear functional of `ζ` (the target wins at the last zero of the gaps it trails). `Q` is split
exactly, along hyperplanes of the tie hyperplane, until the first of these events is the same
on all of it. A linear inequality holds
on a cone if and only if it holds at its rays. If a cap or the target's win comes first, the
sliver ends there; if a neuron flips first, the walk continues in the neighbouring region from
the crossing. Exact logit ties go to the lowest index, as in the exact engine and `np.argmax`.
The result is a set of cones `C` of the tie hyperplane, each with a width `w·ζ` linear in `ζ`.
The Gaussian measure of the sliver `{ζ + τν : ζ ∈ C, 0 ≤ τ ≤ w·ζ}` follows from polar
coordinates in `ζ`: over each ray it is a 2-D wedge, so

    γ(sliver) = c_m ∫_{flat slice of C} Θ_m(w·y/|y|) |y|^{−m} dA,   Θ_m(a) = ∫_0^{atan a} cos^{m−1}t dt,

with `m = n − 1` and `c_m = (2π)^{−(m+1)/2} 2^{(m−1)/2} Γ((m+1)/2)`. The integral is computed by
slice cubature as above (`slice.slice_sliver_measure`). For thin slivers `Θ_m(a) ≈ a` and it
reduces to the first-order formula. The **exact sliver law** sums these measures. Every point
it counts is a wrong input, and no point is counted twice, so it is at most the error. The
difference, the *unattached* errors, consists of the wrong inputs that are not joined to their
nearer wall by a segment along its normal, inside the chamber, on which the model is wrong
throughout. The exact sliver law costs about as much as
the exact error (0.3–24 times as much for 4-input models), so it explains the error rather
than estimating it more cheaply.

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

**The exact sliver law.** Its two parts were checked separately against Monte Carlo. The sliver
measure (`slice.slice_sliver_measure`) agrees with direct sampling of random slivers in 2 to 5
dimensions (`test_sliver.py`). The walks agree with an independent estimate that knows nothing
of linear regions (`sliver_mc.py widths`). That estimate samples points of each tie hyperplane,
finds the true width `w` along each normal line by a grid of forward passes and bisection, with
`np.argmax`, and averages the stretch's Gaussian measure `Φ(w) − 1/2`. On the 3- and 4-input
models of `test_sliver.py` the exact sliver law agrees with it within |z| ≤ 2.0, and the sliver
measure with direct sampling within |z| ≤ 2.3. The code was then reviewed adversarially, as
above: review agents with different lenses (the mathematics, the code, numerics) and skeptic
agents that tried to refute each finding. Four findings led to fixes in the walk and the sliver
measure:

1. **Tie convention.** On the small models the final hidden state can be entirely zero, so all
   logits tie at 0. The exact engine and `np.argmax` then predict the first class, but the walk
   resolved such ties differently. Together, the four fixes moved `M_{3,3}` from −72% to −9.5%
   and `M_{4,3}` from −12% to −1.6%, mostly through this one, and no model with `d >= 5` by more
   than 0.03% of its error.
2. **A neuron flip at the moment the target wins.** There the gaps can vanish together, and the
   walk ended the sliver where the target only tied; it now continues into the next region.
3. **Unbounded walks.** Without a cap, the walk raised an error as soon as the target could not
   win in the current region, even when a neuron flip would carry the line into a region where
   it does. Flips are now considered first, and it raises only if no event at all remains (every
   law reported for `n >= 3` is capped; for `n = 2` see below).
4. **Wide slivers.** The cubature of `Θ_m` also bisects cones across which `w·u` varies a lot,
   not only cones that are wide.

Three findings were refuted: simultaneous flips of two neurons (harmless in the reviewed code;
after the tie-convention fix they were not, see below), the recursion depth (bounded by the number of regions a line crosses) and overflow of `Θ_m`.

Two more problems were found afterwards, the first by testing and the second by a further
review. First,
`test_sliver.py` put the exact sliver law of `M_{3,3}` 13.7 standard errors below its Monte
Carlo estimate. A point-by-point comparison of the widths located the cause. Lines that enter
the dead region (hidden state 0, all logits 0, class 0 predicted) cross it where a logit gap
vanishes and the last active neuron switches off. These are the same functional in exact
arithmetic, but the walk compared events by normalising their difference by its own norm, so
rounding noise of order 10⁻¹⁷ got a sign of order one, and the walk took a spurious win and
ended the sliver. Differences below 10⁻⁹ of the functionals' size are now treated as zero.
Measured on the zoo models, the noise is at most 10⁻¹⁴ of that size and genuine differences
are at least 10⁻⁷ of it. The fix moved `M_{3,3}` from −9.5% to −1.2%, and `M_{4,3}` and
`M_{6,4}` by 0.03% and 0.01% of their errors. All other models were unchanged. Second, an
adversarial review of that fix (four reviewers, on the semantics of the comparisons, their
numerical thresholds, point-by-point agreement on the zoo models and synthetic stress models,
then a skeptic reproducing each finding) found that when several neurons switch at the
same point of a line, the walk applied them one at a time and judged the gaps in between. On a
toy RNN built so that every last-step neuron and every gap vanish together, this ended slivers
early (z = −11.2 against a Monte Carlo estimate with 2²⁰ normal lines per tie; +0.3 after the
fix; the toy is a check in `test_sliver.py`). All simultaneous switches are now applied first. No zoo model has such a crossing inside a wrong
stretch, and recomputing every 2- to 4-input model changed none by more than 1.3 × 10⁻¹²
relative. The review confirmed two smaller issues. The walk raised an error on 2-input models
whose lines stay wrong to infinity (they have no order caps; such slivers now get infinite
width); no zoo model has such lines. And the Monte Carlo reference itself steps over correct
stretches shorter than its 2% grid, which overstates the width, and breaks exact logit ties by
rounding. On the zoo models only grid misses were found, and rarely: 1 of 6000 lines checked on
`AM_{8,5}` (an upward bias of its Monte Carlo law of about 0.07 standard errors), none of 6000
on each of the other 5-input models, and fewer than 1 in 1000 sampled lines of the 3- and
4-input models checked, whose reported laws come from the walk. It refuted two others: the 10⁻⁹ threshold against an ill-conditioned flip (`M_{8,4}` is
bit-identical for any threshold from 10⁻¹³ to 10⁻⁶) and the tie test's dynamic range (the zoo
models' output rows differ in size by at most a factor of 19).

For `n = 2` the exact sliver law is up to 1.3 × 10⁻⁶ relative (3 × 10⁻¹¹ absolute) above the
exact error, as are the first-order laws. This is the precision of the closed-form arcsine
measures of the reference, not an excess of the law.

## Limitations

* **`M_{16,10}` is still open.** The exact engine refines every linear region, and their number
  grows exponentially with sequence length. Sampling 200k inputs of `M_{16,10}` gives 138k
  distinct activation-pattern sequences, so exact refinement is hopeless there. Its errors are
  still tie slivers, but not thin ones. A Monte Carlo estimate of the exact sliver law (true
  widths along 2¹⁶ normal lines per tie hyperplane, `sliver_mc.py widths`) accounts for 96.8% ±
  0.4% of its error rate of 0.0468, while the first-order law is off by +170% uncapped, −14%
  with the order cap and −22% with the bisector cap. 18% of the error comes from the 1.1% of
  inputs with at most one positive value, 78% of which are wrong: the model's only neuron with
  a large positive input weight (+10.2; the next is +0.3) detects new positive running maxima,
  so nothing ranks negative values. What is missing is a mechanistic way to get the widths of
  these slivers without walking each normal line, i.e. a cheap model of how the hidden state
  bends the logit gap across a sliver. (An earlier version of this bullet said the error was
  not tie-sliver dominated; that holds only for the first-order law.)
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
* **The sliver laws explain trained models; they do not estimate noisy ones cheaply.** They
  describe the errors of the zoo models as trained. Under the weight noise of the evaluation
  protocol the 4-input errors grow 6–40 times (`M_{16,4}`: from 0.17% to 1–7% at σ =
  0.002–0.01), and on 9 such noisy models the order-truncated law is off by +8% to +50%. The
  exact sliver law removes the second-order error but costs about as much as the exact
  computation for 4-input models (0.3–24 times as much), and far more for 5-input models, where
  it is only estimated by Monte Carlo. So neither competes with sampling under that protocol.
  It also leaves the unattached errors unexplained: under 1% of the error for the accurate 3-
  and 4-input models, 0.4–1.2% (Monte Carlo) for the accurate 5-input models, 1.2–2.8% for
  `M_{3,3}`, `M_{4,3}` and `M_{6,4}`, and 5–6% for the weak 5-input models.
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
python test_sliver.py M:8,3 M:16,4               # sliver measure and laws vs their Monte Carlo estimates
./run_tables.sh                                 # exact accuracies (n <= 5), tie-sliver laws, per-root MC, and:
#   python sliver_mc.py table                    #   where the wrong inputs lie (one / two / no slivers)
#   python sliver_mc.py widths-table             #   Monte Carlo exact sliver law of the 5-input models
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
