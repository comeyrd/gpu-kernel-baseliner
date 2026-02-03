#include "hip/hip_runtime.h"
#include "ComputationKernel.hpp"
#include <iostream>
#include <random>
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

void ComputationKernel::cpu(ComputationOutput &output) {
  for (int i = 0; i < m_input.m_N; ++i) {
    output.m_c_host[i] = m_input.m_a_host[i] + m_input.m_b_host[i];
  }
  for (int _ = 0; _ < 50; _++) {
    for (int i = 1; i < m_input.m_N; ++i) {
      output.m_c_host[i] = m_input.m_a_host[i] + output.m_c_host[i] + (m_input.m_b_host[i] * m_input.m_b_host[i]);
    }
  }
}

void ComputationKernel::run(std::shared_ptr<hipStream_t> &stream) {
  computation_kernel<<<m_blocks, m_threads, 0, *stream>>>(m_d_a, m_d_b, m_d_c, m_input.m_N);
}
