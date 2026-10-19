#!/bin/zsh
# Development set (seeds 32-95, original seed-only noise; includes the held-out failures found in
# review): compare the finishing rules tie / cguard / lguard. (Originally run as cguard and tie
# first, then lguard, which was designed after seeing those results; see README.) lguard was then
# frozen for run_test.sh.
cd "$(dirname "$0")"
for rule in tie cguard lguard; do
  for spec in M:32,3 M:16,3; do
    for sigma in 0.002 0.005 0.01; do
      ../.venv/bin/python evaluate.py $spec $sigma 32 64 $rule dev 2>&1 | grep -v Warning | grep -v "c = c"
    done
  done
done
echo DEV_DONE
