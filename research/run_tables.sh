#!/bin/zsh
# Exact accuracies and tie-sliver law tables for every AlgZoo RNN classifier with n <= 5 inputs.
# The exact engine runs root cone by root cone in a process pool with per-root checkpoints in
# results/roots/, so every step is resumable; the 5-input models with d = 16, 32 take the longest.
cd "$(dirname "$0")"
PY=../.venv/bin/python
$PY exact_table.py 2 3 4 2>&1 | grep -v Warning | grep -v "c = c" >| results/exact_table_n234.txt
$PY sliver_table.py 2 3 4 2>&1 | grep -v Warning | grep -v "c = c" >| results/sliver_table_n234.txt
$PY exact_table.py 5 2>&1 | grep -v Warning | grep -v "c = c" >| results/exact_table_n5.txt
$PY sliver_table.py 5 2>&1 | grep -v Warning | grep -v "c = c" >| results/sliver_table_n5.txt
$PY sliver_mc.py table 2>&1 | grep -v Warning  # where the wrong inputs lie (one / two / no slivers)
$PY sliver_mc.py widths-table 2>&1 | grep -v Warning  # Monte Carlo exact sliver law for sliver_table.NO_EXACT
for spec in M:16,5 M:32,5 AM:16,5 AM:32,5; do  # per-root Monte Carlo check of the largest models
  $PY mc_roots.py $spec 2>&1 | grep -v Warning
done
echo TABLES_DONE
