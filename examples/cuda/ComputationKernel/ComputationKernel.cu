#include "ComputationKernel.hpp"
#include <baseliner/managers/BenchmarkCaseManager.hpp>
#include <vector>

__global__ void computation_kernel(int *a, int *b, int *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
  __syncthreads();
  for (int i = 0; i < 50; i++) {
    if (idx > 0 && idx < N) {
      c[idx] = a[idx] + c[idx] + b[idx] * b[idx];
    }
    __syncthreads();
  }
}

void ComputationKernel::run(std::shared_ptr<cudaStream_t> stream) {
  computation_kernel<<<m_blocks, m_threads, 0, *stream>>>(m_d_a, m_d_b, m_d_c, get_input()->m_N);
}
