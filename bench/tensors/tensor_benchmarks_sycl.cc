#define EIGEN_USE_SYCL

#include <SYCL/sycl.hpp>
#include <iostream>

#include "tensor_benchmarks.h"

#define BM_FuncGPU(FUNC)                                       \
  static void BM_##FUNC(int iters, int N) {                    \
    StopBenchmarkTiming();                                     \
    cl::sycl::gpu_selector selector; \
    Eigen::QueueInterface queue(selector);     \
    Eigen::SyclDevice device(&queue);                          \
    BenchmarkSuite<Eigen::SyclDevice, float> suite(device, N); \
    suite.FUNC(iters);                                         \
  }                                                            \
  BENCHMARK_RANGE(BM_##FUNC, 10, 5000);

BM_FuncGPU(broadcasting);
BM_FuncGPU(coeffWiseOp);
