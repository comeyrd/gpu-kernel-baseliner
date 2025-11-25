#include <iostream>
#include <random>
#include <vector>
#include "main.hpp"

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


int main() {
  std::cout << "Cuda Device Manipuation" << std::endl;
  int work_size = 1;
  auto input = Computation::Input(work_size);
  input.generate_random();
  auto output = Computation::Output(work_size);
  auto impl = Computation::GpuImplementation(input);
  auto flusher = Baseliner::Backend::HipBackend::L2Flusher();
  auto timer = Baseliner::Backend::HipBackend::GpuTimer();
  auto blocker = Baseliner::Backend::HipBackend::BlockingKernel();

  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));
  impl.setup();
  timer.start(stream);
  impl.run(stream);
  timer.stop(stream);
  std::cout << "Warmup: " << timer.time_elapsed() << std::endl;

for (int r = 0; r < 10; r++) {
    flusher.flush(stream);
    blocker.block(stream, 1000.0);
    timer.start(stream);
    impl.run(stream);
    timer.stop(stream);
    blocker.unblock();
    std::cout << timer.time_elapsed() << " | ";
  }
  impl.teardown(output);
  std::cout << std::endl;
}

void Computation::GpuImplementation::run(hipStream_t &stream) {
  computation_kernel<<<m_blocks, m_threads, 0, stream>>>(m_d_a, m_d_b, m_d_c, m_input.m_N);
}