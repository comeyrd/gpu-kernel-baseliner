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

#ifndef MATRIXMUL_KERNEL_HPP
#define MATRIXMUL_KERNEL_HPP

#include <baseliner/Kernel.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/backend/cuda/CudaBackend.hpp>
#include <cstddef>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

class MatrixMulInput : public Baseliner::IInput {
public:
  void on_update() override {
    allocate();
  };

  void generate_random() override {
    std::mt19937 gen(get_seed());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // NOLINT

    for (auto &val : m_h_A) {
      val = dist(gen);
    }
    for (auto &val : m_h_B) {
      val = dist(gen);
    }
  };

  explicit MatrixMulInput()
      : Baseliner::IInput() { // NOLINT
    allocate();
  };

  void allocate() override {
    // Apply work_size multiplier to the height of A to increase workload
    m_wA = m_wA_base;
    m_hA = m_hA_base * get_work_size();

    // Inner dimensions must match
    m_hB = m_wA;
    m_wB = m_wB_base;

    m_size_A = m_wA * m_hA;
    m_size_B = m_wB * m_hB;

    m_h_A.resize(m_size_A);
    m_h_B.resize(m_size_B);
  }

  // Default dimensions based on the original main() example
  int m_wA_base = 320; // NOLINT 5 * 2 * 32
  int m_hA_base = 320; // NOLINT
  int m_wB_base = 640; // NOLINT 5 * 4 * 32
  int m_hB_base = 320; // NOLINT

  int m_wA, m_hA, m_wB, m_hB;
  int m_size_A, m_size_B;
  int m_block_size = 16;

  std::vector<float> m_h_A;
  std::vector<float> m_h_B;

protected:
  void register_options() override {
    IInput::register_options();
    add_option("MatrixMulInput", "wA", "Width of Matrix A", m_wA_base);
    add_option("MatrixMulInput", "hA", "Height of Matrix A", m_hA_base);
    add_option("MatrixMulInput", "wB", "Width of Matrix B", m_wB_base);
    add_option("MatrixMulInput", "hB", "Height of Matrix B", m_hB_base);
    add_option("MatrixMulInput", "block_size", "Block size (16 or 32)", m_block_size);
  };
};

class MatrixMulOutput : public Baseliner::IOutput<MatrixMulInput> {
public:
  explicit MatrixMulOutput(std::shared_ptr<const MatrixMulInput> input)
      : Baseliner::IOutput<MatrixMulInput>(input) {
    m_size_C = get_input()->m_hA * get_input()->m_wB;
    m_h_C.resize(m_size_C);
  };

  std::vector<float> m_h_C;
  int m_size_C;

  // Optional: equality check for verification
  auto operator==(const MatrixMulOutput &other) const {
    if (m_size_C != other.m_size_C) {
      return false;
    }
    for (size_t i = 0; i < m_h_C.size(); i++) {
      if (std::abs(m_h_C[i] - other.m_h_C[i]) > 1e-3) {
        return false;
      }
    }
    return true;
  }
  friend auto operator<<(std::ostream &oss, const MatrixMulOutput &thing) -> std::ostream & {
    for (int i = 0; i < thing.m_h_C.size(); i++) {
      oss << thing.m_h_C[i] << ", ";
    }
    oss << std::endl;
    return oss;
  }
};

class MatrixMulKernel : public Baseliner::ICudaKernel<MatrixMulInput, MatrixMulOutput> {
public:
  std::string name() override {
    return "MatrixMulKernel";
  };
  void cpu(MatrixMulOutput &output) override;

  void setup() override {
    size_t mem_size_A = get_input()->m_size_A * sizeof(float);
    size_t mem_size_B = get_input()->m_size_B * sizeof(float);
    size_t mem_size_C = get_input()->m_hA * get_input()->m_wB * sizeof(float);

    CHECK_CUDA(cudaMalloc(&m_d_A, mem_size_A));
    CHECK_CUDA(cudaMalloc(&m_d_B, mem_size_B));
    CHECK_CUDA(cudaMalloc(&m_d_C, mem_size_C));

    CHECK_CUDA(cudaMemcpy(m_d_A, get_input()->m_h_A.data(), mem_size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m_d_B, get_input()->m_h_B.data(), mem_size_B, cudaMemcpyHostToDevice));

    // Setup execution parameters
    int block_size = get_input()->m_block_size;
    m_threads = dim3(block_size, block_size);
    m_grid =
        dim3((get_input()->m_wB + m_threads.x - 1) / m_threads.x, (get_input()->m_hA + m_threads.y - 1) / m_threads.y);
  };

  void reset() override {
    // Optional: Zero out C if accumulation logic was involved,
    // but this kernel writes directly (C = ...), so reset isn't strictly necessary
    // unless we want to clear previous results.
    size_t mem_size_C = get_input()->m_hA * get_input()->m_wB * sizeof(float);
    CHECK_CUDA(cudaMemset(m_d_C, 0, mem_size_C));
  };

  void run(std::shared_ptr<cudaStream_t> stream) override;

  void teardown(Output &output) override {
    size_t mem_size_C = get_input()->m_hA * get_input()->m_wB * sizeof(float);
    CHECK_CUDA(cudaMemcpy(output.m_h_C.data(), m_d_C, mem_size_C, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(m_d_A));
    CHECK_CUDA(cudaFree(m_d_B));
    CHECK_CUDA(cudaFree(m_d_C));
  };

  MatrixMulKernel(std::shared_ptr<const MatrixMulInput> input)
      : Baseliner::ICudaKernel<Input, Output>(input) {};

private:
  float *m_d_A = nullptr;
  float *m_d_B = nullptr;
  float *m_d_C = nullptr;
  dim3 m_threads{};
  dim3 m_grid{};
};

#endif // MATRIXMUL_KERNEL_HPP