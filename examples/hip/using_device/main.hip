#include "hip/hip_runtime.h"
#include <baseliner/backend/hip/HipBackend.hpp>
#include <iostream>
#include <random>
#include <vector>
__global__ void computation_kernel(int *a, int *b, int *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
  __syncthreads();

  if (idx > 0 && idx < N) {
    c[idx - 1] = a[idx] + c[idx] + b[idx] * b[idx];
  }
  __syncthreads();
}

void run_kernel() {
  int N = 5000;

  // Host arrays
  std::vector<int> h_a(N);
  std::vector<int> h_b(N);
  std::vector<int> h_c(N);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(1, 100);
  for (int i = 0; i < N; i++) {
    h_a[i] = dist(gen);
    h_b[i] = dist(gen);
    h_c[i] = 0;
  }

  int *d_a, *d_b, *d_c;

  CHECK_HIP(hipMalloc(&d_a, N * sizeof(int)));
  CHECK_HIP(hipMalloc(&d_b, N * sizeof(int)));
  CHECK_HIP(hipMalloc(&d_c, N * sizeof(int)));

  CHECK_HIP(hipMemcpy(d_a, h_a.data(), N * sizeof(int), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(d_b, h_b.data(), N * sizeof(int), hipMemcpyHostToDevice));

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  auto backend = Baseliner::Backend::HipBackend();
  auto stream = backend.create_stream();
  auto flusher = Baseliner::Backend::HipBackend::L2Flusher();
  auto timer = Baseliner::Backend::HipBackend::GpuTimer(stream);
  auto blocker = Baseliner::Backend::HipBackend::BlockingKernel();
  
  timer.start();
  computation_kernel<<<blocksPerGrid, threadsPerBlock, 0, *stream>>>(d_a, d_b, d_c, N);
  timer.stop();
  std::cout << "Warmup: " << timer.time_elapsed().count() << std::endl;

  for (int r = 0; r < 10; r++) {
    flusher.flush(stream);
    blocker.block(stream,1000.0);
    timer.start();
    computation_kernel<<<blocksPerGrid, threadsPerBlock, 0, *stream>>>(d_a, d_b, d_c, N);
    timer.stop();
    blocker.unblock();
    std::cout << timer.time_elapsed().count() << " | ";
  }
  std::cout << std::endl;
  CHECK_HIP(hipMemcpy(h_c.data(), d_c, N * sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipFree(d_a));
  CHECK_HIP(hipFree(d_b));
  CHECK_HIP(hipFree(d_c));
}

int main() {
  std::cout << "Hip Device Manipuation" << std::endl;
  run_kernel();
}
