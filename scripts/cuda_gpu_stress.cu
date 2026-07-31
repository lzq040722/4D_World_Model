#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 256;

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

__global__ void stress_kernel(float *data, size_t count, int iterations) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t i = index; i < count; i += stride) {
    float value = data[i];
#pragma unroll 4
    for (int j = 0; j < iterations; ++j) {
      value = fmaf(value, 1.000000119f, 0.000000119f);
      value = fmaf(value, 0.999999881f, -0.000000059f);
    }
    data[i] = value;
  }
}

int parse_positive(const char *value, const char *name) {
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (*value == '\0' || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
    std::fprintf(stderr, "%s must be a positive integer\\n", name);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(parsed);
}

int parse_nonnegative(const char *value, const char *name) {
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (*value == '\0' || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
    std::fprintf(stderr, "%s must be a non-negative integer\\n", name);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(parsed);
}

void usage(const char *program) {
  std::fprintf(stderr, "Usage: %s --seconds N [--gpu N] [--memory-mib N]\\n", program);
}

}  // namespace

int main(int argc, char **argv) {
  int seconds = 0;
  int gpu = 0;
  int memory_mib = 4096;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--seconds") == 0 && i + 1 < argc) {
      seconds = parse_positive(argv[++i], "--seconds");
    } else if (std::strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
      gpu = parse_nonnegative(argv[++i], "--gpu");
    } else if (std::strcmp(argv[i], "--memory-mib") == 0 && i + 1 < argc) {
      memory_mib = parse_positive(argv[++i], "--memory-mib");
    } else {
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  if (seconds == 0) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (gpu < 0 || gpu >= device_count) {
    std::fprintf(stderr, "GPU index must be between 0 and %d\\n", device_count - 1);
    return EXIT_FAILURE;
  }
  check(cudaSetDevice(gpu), "cudaSetDevice");

  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, gpu), "cudaGetDeviceProperties");
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

  const size_t requested_bytes = static_cast<size_t>(memory_mib) * 1024 * 1024;
  const size_t maximum_bytes = free_bytes * 7 / 10;
  const size_t bytes = std::min(requested_bytes, maximum_bytes);
  if (bytes < 64ULL * 1024 * 1024) {
    std::fprintf(stderr, "Less than 64 MiB of GPU memory is safely available; refusing to run.\\n");
    return EXIT_FAILURE;
  }

  float *data = nullptr;
  check(cudaMalloc(&data, bytes), "cudaMalloc");
  check(cudaMemset(data, 0, bytes), "cudaMemset");

  const size_t count = bytes / sizeof(float);
  const int blocks = std::min(65535, std::max(1, properties.multiProcessorCount * 32));
  std::printf("GPU %d (%s): stressing for %d seconds with %zu MiB of memory\\n", gpu,
              properties.name, seconds, bytes / 1024 / 1024);
  std::fflush(stdout);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  auto last_report = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() < deadline) {
    stress_kernel<<<blocks, kThreads>>>(data, count, 128);
    check(cudaGetLastError(), "stress_kernel launch");
    check(cudaDeviceSynchronize(), "stress_kernel execution");

    const auto now = std::chrono::steady_clock::now();
    if (now - last_report >= std::chrono::seconds(30)) {
      const auto remaining = std::chrono::duration_cast<std::chrono::seconds>(deadline - now).count();
      std::printf("GPU %d stress test running; %lld seconds remaining\\n", gpu,
                  static_cast<long long>(std::max<int64_t>(0, remaining)));
      std::fflush(stdout);
      last_report = now;
    }
  }

  check(cudaFree(data), "cudaFree");
  std::puts("GPU stress test completed successfully.");
  return EXIT_SUCCESS;
}
