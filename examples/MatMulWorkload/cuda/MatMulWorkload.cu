/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Copyright 2026, Come Eyraud.

#include "../MatMulWorkload.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int aBegin = wA * BLOCK_SIZE * by;
  int aEnd = aBegin + wA - 1;
  int aStep = BLOCK_SIZE;
  int bBegin = BLOCK_SIZE * bx;
  int bStep = BLOCK_SIZE * wB;

  float Csub = 0;

  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}

// ---------------------------------------------------------------------------
// CUDA specializations
// ---------------------------------------------------------------------------

template <>
void MatrixMulWorkload<Baseliner::Hardware::CudaBackend>::setup_device(
    std::shared_ptr<typename backend::stream_t> stream) {
  size_t mem_size_A = m_size_A * sizeof(float);
  size_t mem_size_B = m_size_B * sizeof(float);
  size_t mem_size_C = m_hA * m_wB * sizeof(float);

  CHECK_CUDA(cudaMalloc(&m_d_A, mem_size_A));
  CHECK_CUDA(cudaMalloc(&m_d_B, mem_size_B));
  CHECK_CUDA(cudaMalloc(&m_d_C, mem_size_C));

  CHECK_CUDA(cudaMemcpyAsync(m_d_A, m_h_A.data(), mem_size_A, cudaMemcpyHostToDevice, *stream));
  CHECK_CUDA(cudaMemcpyAsync(m_d_B, m_h_B.data(), mem_size_B, cudaMemcpyHostToDevice, *stream));
  m_threads_x = m_block_size;
  m_threads_y = m_block_size;
  m_grid_x = (m_wB + m_threads_x - 1) / m_threads_x;
  m_grid_y = (m_hA + m_threads_y - 1) / m_threads_y;
}

template <>
void MatrixMulWorkload<Baseliner::Hardware::CudaBackend>::reset_device(
    std::shared_ptr<typename backend::stream_t> stream) {
  size_t mem_size_C = m_hA * m_wB * sizeof(float);
  CHECK_CUDA(cudaMemsetAsync(m_d_C, 0, mem_size_C, *stream));
}

template <>
auto MatrixMulWorkload<Baseliner::Hardware::CudaBackend>::run(std::shared_ptr<typename backend::stream_t> stream)
    -> std::monostate {
  if (m_block_size == 16) { // NOLINT
    MatrixMulCUDA<16><<<dim3(m_grid_x, m_grid_y), dim3(m_threads_x, m_threads_y), 0, *stream>>>(m_d_C, m_d_A, m_d_B,
                                                                                                m_wA, m_wB); // NOLINT
  } else {
    MatrixMulCUDA<32><<<dim3(m_grid_x, m_grid_y), dim3(m_threads_x, m_threads_y), 0, *stream>>>(m_d_C, m_d_A, m_d_B,
                                                                                                m_wA, m_wB); // NOLINT
  }
  return {};
}

template <>
void MatrixMulWorkload<Baseliner::Hardware::CudaBackend>::fetch_results(
    std::shared_ptr<typename backend::stream_t> stream) {
  size_t mem_size_C = m_hA * m_wB * sizeof(float);
  CHECK_CUDA(cudaMemcpyAsync(m_h_C.data(), m_d_C, mem_size_C, cudaMemcpyDeviceToHost, *stream));
  CHECK_CUDA(cudaStreamSynchronize(*stream));
  CHECK_CUDA(cudaFree(m_d_A));
  CHECK_CUDA(cudaFree(m_d_B));
  CHECK_CUDA(cudaFree(m_d_C));
  m_d_A = nullptr;
  m_d_B = nullptr;
  m_d_C = nullptr;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

namespace {
  using MatrixMulWorkload = MatrixMulWorkload<Baseliner::Hardware::CudaBackend>;
  BASELINER_REGISTER_WORKLOAD(MatrixMulWorkload);
} // namespace
