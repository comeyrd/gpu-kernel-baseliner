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
  std::cout << "Cuda Kernel Manipuation" << std::endl;
  int work_size = 1;
  auto input = ComputationKernel::Input(work_size);
  input.generate_random();
  auto output = ComputationKernel::Output(work_size);
  auto impl = ComputationKernel(input);
  auto backend = Baseliner::Backend::CudaBackend();
  auto stream = backend.create_stream();
  auto flusher = Baseliner::Backend::CudaBackend::L2Flusher();
  auto timer = Baseliner::Backend::CudaBackend::GpuTimer(stream);
  auto blocker = Baseliner::Backend::CudaBackend::BlockingKernel();

  impl.setup();
  timer.start();
  impl.run(stream);
  timer.stop();
  std::cout << "Warmup: " << timer.time_elapsed().count() << std::endl;

for (int r = 0; r < 10; r++) {
    flusher.flush(stream);
    blocker.block(stream, 1000.0);
    timer.start();
    impl.run(stream);
    timer.stop();
    blocker.unblock();
    std::cout << timer.time_elapsed().count() << " | ";
  }
  impl.teardown(output);
  std::cout << std::endl;
}

void ComputationKernel::run(std::shared_ptr<cudaStream_t> &stream) {
  computation_kernel<<<m_blocks, m_threads, 0, *stream>>>(m_d_a, m_d_b, m_d_c, m_input.m_N);
}