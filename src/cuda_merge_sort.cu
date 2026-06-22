// Copyright 2026. Released under the MIT License (see LICENSE).
//
// GPU Specialization Capstone: CUDA Parallel Merge Sort at Scale.
//
// This program implements a bottom-up, iterative merge sort that runs on the
// GPU. Starting from runs of length one, each pass doubles the run width and
// merges adjacent sorted runs in parallel -- one CUDA thread per output run --
// using two ping-pong device buffers. It can:
//
//   * sort a randomly generated array of a chosen size,
//   * read an array from a CSV file and write the sorted result to CSV,
//   * verify correctness against the C++ standard library sort, and
//   * benchmark GPU vs. CPU sorting across a range of input sizes.
//
// The work is GPU-bound: all merging happens in the MergePassKernel; the CPU
// only orchestrates passes, performs I/O, and provides a reference sort for
// verification and speedup comparison.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                            \
    if (status != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                   cudaGetErrorString(status), __FILE__, __LINE__);         \
      std::exit(EXIT_FAILURE);                                              \
    }                                                                        \
  } while (0)

// Timing breakdown (milliseconds) for a single GPU sort.
struct Timing {
  float h2d_ms = 0.0f;     // Host-to-device copy.
  float kernel_ms = 0.0f;  // All merge passes.
  float d2h_ms = 0.0f;     // Device-to-host copy.
  float total_ms = 0.0f;   // End-to-end on the device side.
};

// Options parsed from the command line.
struct Config {
  long size = 1L << 20;  // Array length when generating random data.
  std::string input;     // If set, read the array from this CSV instead.
  std::string output;    // If set, write the sorted array to this CSV.
  std::string save_input;  // If set, save the generated input array to CSV.
  int threads_per_block = 256;
  unsigned long seed = 42;
  bool benchmark = false;
  std::string bench_out = "proof/benchmark.csv";
  bool verify = true;
};

// ---------------------------------------------------------------------------
// Device merge logic.
// ---------------------------------------------------------------------------

// Merges the two adjacent sorted runs src[start, middle) and src[middle, end)
// into dst[start, end). Stable: equal keys keep their relative order.
__device__ void BottomUpMerge(const long* src, long* dst, long start,
                              long middle, long end) {
  long i = start;
  long j = middle;
  for (long k = start; k < end; ++k) {
    if (i < middle && (j >= end || src[i] <= src[j])) {
      dst[k] = src[i++];
    } else {
      dst[k] = src[j++];
    }
  }
}

// One thread merges one output run of length `width` (two input runs of length
// width/2). Threads whose run starts past the end of the array exit early.
__global__ void MergePassKernel(const long* src, long* dst, long n,
                                long width) {
  const long segment = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long start = segment * width;
  if (start >= n) {
    return;
  }
  long middle = start + width / 2;
  long end = start + width;
  if (middle > n) middle = n;
  if (end > n) end = n;
  BottomUpMerge(src, dst, start, middle, end);
}

// ---------------------------------------------------------------------------
// GPU sort driver.
// ---------------------------------------------------------------------------

// Sorts `input` on the GPU and returns the sorted array. Fills `timing` with a
// breakdown of transfer and kernel time.
std::vector<long> GpuMergeSort(const std::vector<long>& input,
                               int threads_per_block, Timing* timing) {
  const long n = static_cast<long>(input.size());
  std::vector<long> output(input.size());
  if (n <= 1) {
    output = input;
    return output;
  }

  long* d_a = nullptr;
  long* d_b = nullptr;
  const size_t bytes = static_cast<size_t>(n) * sizeof(long);
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));

  cudaEvent_t e_start, e_h2d, e_kernels, e_end;
  CUDA_CHECK(cudaEventCreate(&e_start));
  CUDA_CHECK(cudaEventCreate(&e_h2d));
  CUDA_CHECK(cudaEventCreate(&e_kernels));
  CUDA_CHECK(cudaEventCreate(&e_end));

  CUDA_CHECK(cudaEventRecord(e_start));
  CUDA_CHECK(cudaMemcpy(d_a, input.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(e_h2d));

  // Ping-pong between the two buffers, one merge pass per doubling of width.
  long* src = d_a;
  long* dst = d_b;
  for (long width = 2; width < 2 * n; width <<= 1) {
    const long segments = (n + width - 1) / width;
    const int blocks = static_cast<int>(
        (segments + threads_per_block - 1) / threads_per_block);
    MergePassKernel<<<blocks, threads_per_block>>>(src, dst, n, width);
    CUDA_CHECK(cudaGetLastError());
    std::swap(src, dst);  // `src` now holds the most recently merged data.
  }
  CUDA_CHECK(cudaEventRecord(e_kernels));

  // The final sorted data lives in `src` regardless of pass count -- the swap
  // above keeps `src` pointing at the latest output. (Forgetting this is the
  // classic "result is in the swap buffer" bug for odd pass counts.)
  CUDA_CHECK(cudaMemcpy(output.data(), src, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(e_end));
  CUDA_CHECK(cudaEventSynchronize(e_end));

  if (timing != nullptr) {
    CUDA_CHECK(cudaEventElapsedTime(&timing->h2d_ms, e_start, e_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&timing->kernel_ms, e_h2d, e_kernels));
    CUDA_CHECK(cudaEventElapsedTime(&timing->d2h_ms, e_kernels, e_end));
    CUDA_CHECK(cudaEventElapsedTime(&timing->total_ms, e_start, e_end));
  }

  cudaEventDestroy(e_start);
  cudaEventDestroy(e_h2d);
  cudaEventDestroy(e_kernels);
  cudaEventDestroy(e_end);
  cudaFree(d_a);
  cudaFree(d_b);
  return output;
}

// ---------------------------------------------------------------------------
// Host helpers.
// ---------------------------------------------------------------------------

std::vector<long> GenerateRandomArray(long n, unsigned long seed) {
  std::vector<long> data(static_cast<size_t>(n));
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<long> dist(0, 1000000000L);
  for (long i = 0; i < n; ++i) {
    data[static_cast<size_t>(i)] = dist(rng);
  }
  return data;
}

std::vector<long> ReadCsv(const std::string& path) {
  std::vector<long> data;
  std::ifstream file(path);
  if (!file.is_open()) {
    std::fprintf(stderr, "Could not open input CSV: %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::string token;
  while (std::getline(file, token, ',')) {
    // A token may carry a trailing newline; stol ignores surrounding space.
    if (token.find_first_not_of(" \t\r\n") == std::string::npos) {
      continue;
    }
    data.push_back(std::stol(token));
  }
  return data;
}

void WriteCsv(const std::string& path, const std::vector<long>& data) {
  std::ofstream file(path);
  if (!file.is_open()) {
    std::fprintf(stderr, "Could not open output CSV: %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < data.size(); ++i) {
    file << data[i];
    if (i + 1 < data.size()) {
      file << ',';
    }
  }
  file << '\n';
}

// Times std::sort on a copy of the data; returns milliseconds.
double CpuSort(const std::vector<long>& input, std::vector<long>* sorted) {
  *sorted = input;
  const auto t0 = std::chrono::high_resolution_clock::now();
  std::sort(sorted->begin(), sorted->end());
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// Modes.
// ---------------------------------------------------------------------------

int RunSingle(const Config& config) {
  std::vector<long> input;
  if (!config.input.empty()) {
    input = ReadCsv(config.input);
    std::printf("Loaded %zu values from %s\n", input.size(),
                config.input.c_str());
  } else {
    input = GenerateRandomArray(config.size, config.seed);
    std::printf("Generated %zu random values (seed %lu)\n", input.size(),
                config.seed);
  }
  if (!config.save_input.empty()) {
    WriteCsv(config.save_input, input);
    std::printf("Saved input array to %s\n", config.save_input.c_str());
  }

  Timing timing;
  std::vector<long> gpu_sorted =
      GpuMergeSort(input, config.threads_per_block, &timing);

  std::printf("GPU sort: total %.2f ms (h2d %.2f, kernels %.2f, d2h %.2f)\n",
              timing.total_ms, timing.h2d_ms, timing.kernel_ms, timing.d2h_ms);

  if (config.verify) {
    std::vector<long> cpu_sorted;
    const double cpu_ms = CpuSort(input, &cpu_sorted);
    const bool correct = (gpu_sorted == cpu_sorted);
    std::printf("CPU std::sort: %.2f ms | GPU vs CPU speedup: %.2fx\n", cpu_ms,
                timing.total_ms > 0 ? cpu_ms / timing.total_ms : 0.0);
    std::printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    if (!correct) {
      return EXIT_FAILURE;
    }
  }

  if (!config.output.empty()) {
    WriteCsv(config.output, gpu_sorted);
    std::printf("Wrote sorted output to %s\n", config.output.c_str());
  }
  return EXIT_SUCCESS;
}

int RunBenchmark(const Config& config) {
  const long sizes[] = {1L << 12, 1L << 14, 1L << 16,
                        1L << 18, 1L << 20, 1L << 22};
  std::ofstream csv(config.bench_out);
  if (!csv.is_open()) {
    std::fprintf(stderr, "Could not open benchmark output: %s\n",
                 config.bench_out.c_str());
    return EXIT_FAILURE;
  }
  csv << "size,cpu_ms,gpu_ms,speedup,correct\n";
  std::printf("%12s %12s %12s %10s %8s\n", "size", "cpu_ms", "gpu_ms",
              "speedup", "correct");
  std::printf("---------------------------------------------------------\n");

  for (long n : sizes) {
    std::vector<long> input = GenerateRandomArray(n, config.seed);

    std::vector<long> cpu_sorted;
    const double cpu_ms = CpuSort(input, &cpu_sorted);

    Timing timing;
    std::vector<long> gpu_sorted =
        GpuMergeSort(input, config.threads_per_block, &timing);

    const bool correct = (gpu_sorted == cpu_sorted);
    const double speedup =
        timing.total_ms > 0 ? cpu_ms / timing.total_ms : 0.0;

    std::printf("%12ld %12.2f %12.2f %9.2fx %8s\n", n, cpu_ms, timing.total_ms,
                speedup, correct ? "yes" : "NO");
    csv << n << ',' << cpu_ms << ',' << timing.total_ms << ',' << speedup << ','
        << (correct ? "yes" : "no") << '\n';
  }
  std::printf("---------------------------------------------------------\n");
  std::printf("Benchmark results written to %s\n", config.bench_out.c_str());
  return EXIT_SUCCESS;
}

// ---------------------------------------------------------------------------
// Command line.
// ---------------------------------------------------------------------------

void PrintUsage(const char* program) {
  std::printf(
      "Usage: %s [options]\n"
      "  --size <N>          Random array length (default 1048576)\n"
      "  --input <csv>       Read the array from a CSV file instead\n"
      "  --output <csv>      Write the sorted array to a CSV file\n"
      "  --save-input <csv>  Save the generated input array to a CSV file\n"
      "  --block <N>         Threads per block (default 256)\n"
      "  --seed <N>          Random seed (default 42)\n"
      "  --benchmark         Sweep sizes, compare GPU vs CPU, write CSV\n"
      "  --bench-out <csv>   Benchmark output path (default proof/benchmark.csv)\n"
      "  --no-verify         Skip CPU verification\n"
      "  --help              Show this message\n",
      program);
}

bool ParseArgs(int argc, char** argv, Config* config) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = (i + 1 < argc);
    if (arg == "--help") {
      PrintUsage(argv[0]);
      return false;
    } else if (arg == "--size" && has_value) {
      config->size = std::atol(argv[++i]);
    } else if (arg == "--input" && has_value) {
      config->input = argv[++i];
    } else if (arg == "--output" && has_value) {
      config->output = argv[++i];
    } else if (arg == "--save-input" && has_value) {
      config->save_input = argv[++i];
    } else if (arg == "--block" && has_value) {
      config->threads_per_block = std::atoi(argv[++i]);
    } else if (arg == "--seed" && has_value) {
      config->seed = std::strtoul(argv[++i], nullptr, 10);
    } else if (arg == "--benchmark") {
      config->benchmark = true;
    } else if (arg == "--bench-out" && has_value) {
      config->bench_out = argv[++i];
    } else if (arg == "--no-verify") {
      config->verify = false;
    } else {
      std::fprintf(stderr, "Unknown or incomplete argument: %s\n", arg.c_str());
      PrintUsage(argv[0]);
      return false;
    }
  }
  return true;
}

void PrintGpuInfo() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int driver = 0, runtime = 0;
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  std::printf("GPU device         : %s (%d SMs)\n", prop.name,
              prop.multiProcessorCount);
  std::printf("CUDA driver/runtime: %d.%d / %d.%d\n", driver / 1000,
              (driver % 100) / 10, runtime / 1000, (runtime % 100) / 10);
}

}  // namespace

int main(int argc, char** argv) {
  Config config;
  if (!ParseArgs(argc, argv, &config)) {
    return EXIT_SUCCESS;
  }
  PrintGpuInfo();
  std::printf("Threads per block  : %d\n", config.threads_per_block);
  std::printf("----------------------------------------------------------\n");

  return config.benchmark ? RunBenchmark(config) : RunSingle(config);
}
