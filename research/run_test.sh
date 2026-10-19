#!/bin/zsh
# Final held-out test: fresh seeds 1000-1199, noise keyed by (seed, sigma), frozen rule lguard.
# Resumable: re-running skips noisy models already in results/test/.
cd "$(dirname "$0")"
for spec in M:32,3 M:16,3 M:8,3 M:6,3 M:5,3 M:4,3 M:3,3; do
  for sigma in 0.002 0.005 0.01; do
    ../.venv/bin/python evaluate.py $spec $sigma 1000 200 lguard test 2>&1 | grep -v Warning | grep -v "c = c"
  done
done
echo TEST_DONE
