#!/usr/bin/env bash
#
# End-to-end driver for the CUDA Parallel Merge Sort capstone.
#
#   1. Builds the binary with make.
#   2. Sorts a large random array, saving input and output CSV files.
#   3. Runs the GPU-vs-CPU scaling benchmark, writing proof/benchmark.csv.
#   4. Captures all console output to proof/run_log.txt.
#
# Usage:
#   ./run.sh            # default 1,048,576-element sort + full benchmark
#   ./run.sh 4194304    # choose the single-run array size

set -euo pipefail
cd "$(dirname "$0")"

SIZE="${1:-1048576}"
LOG="proof/run_log.txt"
mkdir -p proof

echo "==> Building project"
make

echo "==> Single large sort (size=$SIZE) with CSV input/output" | tee "$LOG"
./bin/cuda_merge_sort \
  --size "$SIZE" \
  --save-input "proof/input_${SIZE}.csv" \
  --output "proof/sorted_${SIZE}.csv" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==> Small sample sort (committed to repo as readable proof)" | tee -a "$LOG"
./bin/cuda_merge_sort \
  --size 256 \
  --save-input proof/sample_input.csv \
  --output proof/sample_sorted.csv | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==> GPU vs CPU scaling benchmark" | tee -a "$LOG"
./bin/cuda_merge_sort --benchmark --bench-out proof/benchmark.csv | tee -a "$LOG"

echo "==> Done. Artifacts in proof/:"
ls -1 proof/
