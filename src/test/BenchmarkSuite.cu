#include "unitTests.h"
#include <chrono>
#include <vector>


void registerBenchmarkTests() {
  // Dummy function to force linker to include this object file
  volatile int x = 1;
}

template <typename T> struct BenchmarkTest : public testing::Test {
  using ParamType = T;
};

// Use uint64_t for fixed-point arithmetic (standard in Piranha)
TYPED_TEST_CASE(BenchmarkTest, uint64_t);

// --- VECTOR BENCHMARKS ---

TYPED_TEST(BenchmarkTest, BenchmarkSum) {
  using T = typename TestFixture::ParamType;
  if (partyNum >= TPC<T>::numParties)
    return;

  // Sizes to test (Scaling)
  std::vector<int> sizes = {100, 500, 1000, 2000};

  for (int N : sizes) {
    if (piranha_config["debug_print"])
      std::cout << "Running Sum Benchmark N=" << N << std::endl;

    TPC<T> a(N);
    TPC<T> b(N);
    TPC<T> c(N); // temp for sum
    TPC<T> result(1);

    a.fill(1);
    b.fill(2);

    // Warmup
    sum_and_reduce(a, b, c, result);

    func_profiler.start();
    for (int i = 0; i < 10; i++) { // Run 10 times
      sum_and_reduce(a, b, c, result);
    }
    cudaDeviceSynchronize();

    if (partyNum == 0) {
      // Use manual timing if profiler isn't granular enough, but for now we
      // assume consistent profiler usage Note: func_profiler typically
      // accumulates. We print the average.
      double time_ms = func_profiler.get_elapsed("sum_and_reduce") / 10.0;
      // If get_elapsed returns 0 (because sum_and_reduce isn't instrumented),
      // we print a fallback
      printf("Benchmark_Sum_N%d_AvgTime: %f ms\n", N, time_ms);
      func_profiler.clear(); // Reset for next size
    }
  }
}

TYPED_TEST(BenchmarkTest, BenchmarkCount) {
  using T = typename TestFixture::ParamType;
  if (partyNum >= TPC<T>::numParties)
    return;

  std::vector<int> sizes = {100, 500, 1000, 2000};
  TPC<T> threshold(1);
  threshold.fill(50);

  for (int N : sizes) {
    TPC<T> input(N);
    TPC<T> result(1);
    input.fill(51);

    TPC<T> thresholdExpanded(N);
    thresholdExpanded.fill(50);

    auto start = std::chrono::high_resolution_clock::now();
    count_gt(input, thresholdExpanded, result);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    if (partyNum == 0) {
      printf("Benchmark_Count_N%d_Time: %f s\n", N, diff.count());
    }
  }
}

TYPED_TEST(BenchmarkTest, BenchmarkReLU) {
  using T = typename TestFixture::ParamType;
  if (partyNum >= TPC<T>::numParties)
    return;

  std::vector<int> sizes = {100, 500, 1000, 2000};

  for (int N : sizes) {
    TPC<T> input(N);
    TPC<T> result(N);
    TPC<uint8_t> dresult(N);

    input.fill(-5);

    auto start = std::chrono::high_resolution_clock::now();
    ReLU(input, result, dresult);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    if (partyNum == 0) {
      printf("Benchmark_ReLU_N%d_Time: %f s\n", N, diff.count());
    }
  }
}

TYPED_TEST(BenchmarkTest, BenchmarkBillionaire) {
  using T = typename TestFixture::ParamType;
  if (partyNum >= TPC<T>::numParties)
    return;

  std::vector<int> sizes = {100, 500, 1000, 2000};

  for (int N : sizes) {
    TPC<T> a_cash(N), a_prop(N), a_stock(N);
    TPC<T> b_cash(N), b_prop(N), b_stock(N);
    TPC<T> result(1);

    a_cash.fill(100);
    b_cash.fill(50);

    auto start = std::chrono::high_resolution_clock::now();
    billionaire(a_cash, a_prop, a_stock, b_cash, b_prop, b_stock, result);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    if (partyNum == 0)
      printf("Benchmark_Billionaire_N%d_Time: %f s\n", N, diff.count());
  }
}

// --- MATRIX BENCHMARKS ---

TYPED_TEST(BenchmarkTest, BenchmarkMatMul) {
  using T = typename TestFixture::ParamType;
  if (partyNum >= TPC<T>::numParties)
    return;

  std::vector<int> sizes = {8, 16, 32};

  for (int N : sizes) {
    TPC<T> A(N * N);
    TPC<T> B(N * N);
    TPC<T> C(N * N);

    A.fill(1);
    B.fill(1);

    auto start = std::chrono::high_resolution_clock::now();
    matmul(A, B, C, N, N, N, false, false, false, (T)0);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    if (partyNum == 0)
      printf("Benchmark_MatMul_%dx%d_Time: %f s\n", N, N, diff.count());
  }
}

TYPED_TEST(BenchmarkTest, BenchmarkConv2D) {
  using T = typename TestFixture::ParamType;
  if (partyNum >= TPC<T>::numParties)
    return;

  std::vector<int> sizes = {8, 16, 32};
  int K = 3;

  for (int N : sizes) {
    TPC<T> Image(N * N);
    TPC<T> Filter(K * K);
    TPC<T> Output(N * N);

    Image.fill(1);
    Filter.fill(1);

    auto start = std::chrono::high_resolution_clock::now();
    convolution(Image, Filter, Output, cutlass::conv::Operator::kFprop, 1, N, N,
                K, 1, 1, 1, 1, 0);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    if (partyNum == 0)
      printf("Benchmark_Conv2D_%dx%d_Time: %f s\n", N, N, diff.count());
  }
}
