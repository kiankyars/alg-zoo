#!/bin/bash
# Pre-registered held-out test of the lean ladder (PREREG_lean.md): every 3-input cell, noise seeds
# 7000-7199 keyed by (seed, sigma), both ladders (lean and the published lguard) on the same models,
# results in results/test7000/. Refuses to run on uncommitted code and never mixes seed ranges.
# Resumable: re-running skips noisy models already written.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 REQUIRE_CLEAN=1 WORKERS=${WORKERS:-4}
for spec in M:32,3 M:16,3 M:8,3 M:6,3 M:5,3 M:4,3 M:3,3; do
  for sigma in 0.002 0.005 0.01; do
    for rule in lean lguard; do
      python evaluate.py $spec $sigma 7000 200 $rule test7000 2>&1 | grep -v Warning | grep -v "c = c" || true
    done
  done
done
python mc_validate.py results/test7000/M_*_lguard.jsonl  # independent Monte Carlo check of the truth
python analyze_test.py results/test7000 | tee results/test7000/analysis.txt
echo TEST7000_DONE
