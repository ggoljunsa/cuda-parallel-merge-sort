# CUDA Parallel Merge Sort at Scale — GPU Specialization Capstone

A from-scratch **bottom-up parallel merge sort** that runs on an NVIDIA GPU in
**CUDA C++**, built as my capstone for the GPU Specialization. It sorts arrays
of **millions of elements**, verifies correctness against `std::sort`, and
**benchmarks GPU vs. CPU performance across input sizes** to show where GPU
parallelism actually pays off.

This started from the course's parallel merge-sort lab and was extended into a
complete, configurable tool: CSV input/output, a CLI, correctness verification,
a scaling benchmark that emits a CSV, and full build/run automation. The repo
follows the official course
[project template](https://github.com/PascaleCourseraCourses/CUDAatScaleForTheEnterpriseCourseProjectTemplate)
layout.

---

## The algorithm

Classic merge sort splits an array in half recursively, then merges sorted
pieces back together. The **bottom-up** formulation flips this: it starts with
runs of length 1 (already sorted) and repeatedly merges adjacent runs, doubling
the run width each pass — width 2, 4, 8, … until one run covers the array.

This maps beautifully onto a GPU: within a single pass, **every pair of runs is
independent**, so they can be merged in parallel. For an array of 1,048,576
elements, the first pass merges 524,288 pairs — a CPU does these one after
another, while the GPU assigns **one thread per output run** and does thousands
at once.

```
pass width=2 :  [3,1][4,1][5,9][2,6]  ->  [1,3][1,4][5,9][2,6]
pass width=4 :  [1,3,1,4][5,9,2,6]    ->  [1,1,3,4][2,5,6,9]
pass width=8 :  [1,1,3,4,2,5,6,9]     ->  [1,1,2,3,4,5,6,9]
```

### Kernels and key code

| Component | Role |
|-----------|------|
| `MergePassKernel` | One thread per output run; computes its `[start, middle, end)` and merges. |
| `BottomUpMerge` (device) | The two-pointer stable merge of two sorted runs. |
| `GpuMergeSort` (host) | Allocates two device buffers, loops the passes, **ping-pongs** between buffers, copies the result back, and records CUDA-event timings. |

**Ping-pong buffers.** Two device arrays `d_a`/`d_b` alternate as source and
destination each pass, so no pass overwrites data it is still reading. After the
loop the result is always in `src` because the host swaps the pointers after
every pass — this is what makes the odd-pass case (e.g. 128 elements needs 7
passes) come out correct.

---

## Build

Requires only the CUDA Toolkit (no third-party libraries). See [INSTALL](INSTALL).

```bash
make
# or: make NVCC=/usr/local/cuda/bin/nvcc CUDA_PATH=/usr/local/cuda
```

## Run

One command builds, sorts a large array, runs the benchmark, and logs
everything to `proof/`:

```bash
./run.sh                 # 1,048,576-element sort + full GPU-vs-CPU benchmark
./run.sh 4194304         # pick the single-run size
```

Or drive the binary directly:

```bash
./bin/cuda_merge_sort --size 1048576 --output sorted.csv
./bin/cuda_merge_sort --input my_array.csv --output sorted.csv
./bin/cuda_merge_sort --benchmark --bench-out proof/benchmark.csv
```

### Command-line arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--size <N>` | `1048576` | Random array length to generate |
| `--input <csv>` | — | Read the array from a CSV file instead |
| `--output <csv>` | — | Write the sorted array to a CSV file |
| `--save-input <csv>` | — | Save the generated input array to CSV |
| `--block <N>` | `256` | Threads per block |
| `--seed <N>` | `42` | Random seed |
| `--benchmark` | off | Sweep sizes, compare GPU vs CPU, write CSV |
| `--bench-out <csv>` | `proof/benchmark.csv` | Benchmark output path |
| `--no-verify` | off | Skip CPU verification |
| `--help` | — | Show usage |

### Example output

```
GPU device         : Tesla T4 (40 SMs)
CUDA driver/runtime: 12.2 / 12.2
Threads per block  : 256
----------------------------------------------------------
Generated 1048576 random values (seed 42)
GPU sort: total 9.41 ms (h2d 1.86, kernels 5.93, d2h 1.62)
CPU std::sort: 78.20 ms | GPU vs CPU speedup: 8.31x
Verification: PASSED
```

---

## Proof of execution

`./run.sh` populates `proof/`:

- **`run_log.txt`** — full log: GPU info, single-sort timing breakdown,
  verification result, and the benchmark table.
- **`benchmark.csv`** — `size,cpu_ms,gpu_ms,speedup,correct` across six sizes
  from 4K to 4M elements (the input *and* output data of the experiment).
- **`sample_input.csv` / `sample_sorted.csv`** — a small, human-readable
  before/after pair committed directly to the repo.

The large `input_<N>.csv` / `sorted_<N>.csv` files are produced too but
git-ignored to keep the repo lean.

## Results & lessons learned

- **GPU wins only at scale.** For small arrays (a few thousand elements) the CPU
  `std::sort` is faster — kernel-launch and transfer overhead dominate. The
  crossover appears in the tens-of-thousands range, and by a few million
  elements the GPU is several times faster. The benchmark CSV makes this
  crossover visible and is the most interesting result of the project.
- **Memory traffic, not compute, is the limit.** Each pass reads and writes the
  whole array, so total time tracks bandwidth × number of passes (log₂N). The
  host↔device copies are a real fraction of the end-to-end time, which is why
  the timing is reported as a breakdown.
- **Two bugs I hit and fixed.**
  1. *Output format* — the grader expected comma-separated values; my first
     version printed space-separated, so `WriteCsv` now emits strict CSV.
  2. *Result-in-the-swap-buffer* — for odd pass counts the sorted data ends up
     in the second buffer. An early version freed/copied the wrong buffer and
     returned garbage. The pointer-swap-after-each-pass design (copy from
     `src`) fixes this for any size.

## Next steps

- Overlap host↔device transfers with kernels using CUDA **streams**.
- Use **shared memory** for the first few small-width passes to cut global
  memory traffic.
- Add a `__device__` binary search so a single long run can be merged
  cooperatively by many threads (path-merge), improving load balance on the
  final passes.

## License

MIT — see [LICENSE](LICENSE).
