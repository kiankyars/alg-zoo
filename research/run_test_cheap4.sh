#!/bin/bash
# Pre-registered held-out test of the cheap-measure ladder for 4-input models (PREREG_cheap4.md):
# M_{8,4} and M_{16,4} at three noise levels, noise seeds 8000-8099 keyed by (seed, sigma), both
# ladders (cheap4 and the published lguard) on the same models, results in results/test8000/.
# evaluate.py (REQUIRE_CLEAN) refuses to evaluate uncommitted code; the script stops if it does.
set -e -o pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 REQUIRE_CLEAN=1 WORKERS=${WORKERS:-4}
for spec in M:8,4 M:16,4; do
  for sigma in 0.002 0.005 0.01; do
    for rule in cheap4 lguard; do
      python evaluate.py $spec $sigma 8000 100 $rule test8000 2>&1 | { grep -v Warning | grep -v "c = c" || true; }
    done
  done
done
python mc_validate.py results/test8000/M_*_lguard.jsonl  # independent Monte Carlo check of the truth
python analyze_test4.py results/test8000 | tee results/test8000/analysis.txt
echo TEST8000_DONE
