#include "main.hpp"
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

void ComputationKernel::cpu(ComputationOutput &output) {
  for (int i = 0; i < m_input.m_N; ++i) {
    output.m_c_host[i] = m_input.m_a_host[i] + m_input.m_b_host[i];
  }
  for (int i = 1; i < m_input.m_N; ++i) {
    output.m_c_host[i - 1] = m_input.m_a_host[i] + output.m_c_host[i] + (m_input.m_b_host[i] * m_input.m_b_host[i]);
  }
}

int main() {
  std::cout << "Cuda Kernel Manipuation" << std::endl;
  auto input = ComputationKernel::Input();
  input.generate_random();
  auto output = ComputationKernel::Output(input);
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