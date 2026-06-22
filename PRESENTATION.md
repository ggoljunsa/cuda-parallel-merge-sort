# Presentation Script — CUDA Parallel Merge Sort (5–10 min)

A recording script for the capstone demonstration video. Keep the GitHub repo,
the source file `src/cuda_merge_sort.cu`, and a terminal running `./run.sh`
visible and switch between them on the **[SCREEN]** cues. Target length ≈ 7 min.

---

### [0:00] Introduction (40s)
> **[SCREEN]** GitHub repo front page / README.

"Hi, my name is [YOUR NAME], and this is my GPU Specialization capstone:
**a parallel merge sort implemented in CUDA C++** that runs on an NVIDIA GPU.
The goal was to take a classic sorting algorithm, express it so that hundreds of
GPU threads cooperate, and then measure how its performance compares to a CPU as
the data grows to millions of elements. Let me walk through the idea, the code,
a live run, and what I learned."

---

### [0:40] The algorithm and why it fits a GPU (1m20s)
> **[SCREEN]** README diagram of the bottom-up passes.

"Normal merge sort recursively splits an array and merges the pieces back. I
used the **bottom-up** version instead: I start by treating every single element
as a sorted run of length one, then I repeatedly merge neighboring runs, doubling
the run width each pass — width 2, then 4, then 8, and so on until one run covers
the whole array.

The key insight is this: **within one pass, every pair of runs is completely
independent.** For a million-element array, the first pass merges over five
hundred thousand pairs. A CPU does those one at a time. On the GPU I assign
**one thread to each output run**, so thousands of merges happen at the same
time. That's the parallelism this algorithm exposes, and it's why merge sort is
a natural fit for the GPU."

---

### [2:00] Code walkthrough (1m40s)
> **[SCREEN]** `src/cuda_merge_sort.cu` — scroll to `MergePassKernel` and `BottomUpMerge`.

"Here's the core. `MergePassKernel` is the GPU kernel. Each thread computes a
segment index from its block and thread ID, works out the `start`, `middle`, and
`end` of the two runs it's responsible for, and calls `BottomUpMerge` — a simple
two-pointer stable merge that writes the combined run into the destination
buffer.

> **[SCREEN]** scroll to `GpuMergeSort`.

On the host side, `GpuMergeSort` allocates **two device buffers** and
**ping-pongs** between them: one pass reads from buffer A and writes to B, the
next reads B and writes A. That way a pass never overwrites data it's still
reading. I record CUDA events around the copies and the kernels so I can report
exactly where the time goes — host-to-device, kernels, and device-to-host
separately. Notice there are no third-party libraries here — this is pure CUDA
and the standard library."

---

### [3:40] Live demo (1m40s)
> **[SCREEN]** Terminal: run `./run.sh`.

"Let me run it. `run.sh` builds the project, sorts a one-million-element array,
and then runs a benchmark.

> **[SCREEN]** point at the single-sort output.

For the single sort, you can see the GPU device it detected, the timing
breakdown, and — importantly — **'Verification: PASSED'**: I sort the same data
with `std::sort` and confirm the GPU result is identical, element for element.

> **[SCREEN]** point at the benchmark table, then open `proof/benchmark.csv`.

Then the benchmark sweeps six sizes from four thousand up to four million
elements and writes this CSV with the CPU time, GPU time, and speedup for each.
Every row is verified correct. These CSV files and the log are my proof of
execution, committed to the repo."

---

### [5:20] Results — the interesting part (1m)
> **[SCREEN]** benchmark.csv / a quick chart if you made one.

"Here's the most interesting finding. For **small arrays the CPU is actually
faster** — the cost of launching kernels and copying data over the PCIe bus
outweighs the parallelism. But as the array grows, the GPU pulls ahead, and by a
few million elements it's several times faster. So the lesson isn't 'GPUs are
always faster' — it's that **GPUs win once the problem is big enough to amortize
their overhead.** I also found the runtime is limited by **memory bandwidth**,
not compute: every pass reads and writes the whole array, so the time scales with
bandwidth times log-N passes."

---

### [6:20] Challenges & next steps (40s)
> **[SCREEN]** README "lessons learned" section.

"Two bugs taught me the most. First, an output-format mismatch — the grader
wanted comma-separated values and I'd printed spaces. Second, and more subtly,
for an **odd number of passes the sorted data ends up in the second buffer**;
an early version copied back the wrong one and returned garbage. Fixing that with
a clean pointer-swap design is what makes it correct for any size.

Next, I'd like to overlap transfers with computation using CUDA streams, and use
shared memory for the early passes to cut global-memory traffic."

---

### [7:00] Closing (15s)
> **[SCREEN]** repo URL.

"All the code, build files, and proof artifacts are in the public repository on
screen. Thanks for watching."

---

## Recording tips
- Do a dry run of `./run.sh` first so the build is cached and the live run is
  quick on camera.
- If the benchmark is slow to narrate over, run it once beforehand and just show
  the resulting `benchmark.csv`.
- Stay above 5:00 (the rubric floor) and under 10:00; this script lands ~7 min.
- For the "Excellent" tier, spend a few extra seconds on the next-steps slide —
  the rubric rewards discussing future work.
