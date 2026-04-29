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

#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <baseliner/core/Workload.hpp>

#include <cmath>
#include <ostream>
#include <random>
#include <string>
#include <vector>

template <typename BackendT>
class MatrixMulWorkload : public Baseliner::IWorkload<BackendT> {
public:
  using backend = typename MatrixMulWorkload::backend;

  MatrixMulWorkload() = default;

  // Identification
  auto algo() -> std::string override {
    return "MatrixMul";
  }

  // Metrics
  auto number_of_floating_point_operations() -> std::optional<size_t> override {
    return 2ULL * m_hA * m_wA * m_wB;
  }
  auto number_of_bytes() -> std::optional<size_t> override {
    size_t elements_A = m_hA * m_wA;
    size_t elements_B = m_hB * m_wB;
    size_t elements_C = m_hA * m_wB;
    return (elements_A + elements_B + elements_C) * sizeof(float);
  }

  // Host setup
  void setup_host_random_generated() override {
    compute_dimensions();
    allocate_host_buffers();

    std::mt19937 gen(this->get_seed());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // NOLINT
    for (auto &val : m_h_A) {
      val = dist(gen);
    }
    for (auto &val : m_h_B) {
      val = dist(gen);
    }
  }

  void setup_host_from_file(std::string &path) override {
    compute_dimensions();
    allocate_host_buffers();
    // TODO: deserialize m_h_A, m_h_B, m_h_C_reference from file at path
  }

  void inner_save_setup(std::string &path) override {};

  void results_from_reference() override {
    m_h_C_reference.assign(m_hA * m_wB, 0.0f);
    for (int i = 0; i < m_hA; i++) {
      for (int j = 0; j < m_wB; j++) {
        float sum = 0.0f;
        for (int k = 0; k < m_wA; k++) {
          sum += m_h_A[i * m_wA + k] * m_h_B[k * m_wB + j];
        }
        m_h_C_reference[i * m_wB + j] = sum;
      }
    }
  }

  void setup_device(std::shared_ptr<typename backend::stream_t> stream) override;
  void reset_device(std::shared_ptr<typename backend::stream_t> stream) override;
  auto run(std::shared_ptr<typename backend::stream_t> stream) -> typename backend::launch_result_t override;
  void fetch_results(std::shared_ptr<typename backend::stream_t> stream) override;
  void free() override {
    m_h_A.clear();
    m_h_B.clear();
    m_h_C.clear();
    m_h_C_reference.clear();
  }

  // Validation
  auto validate() -> bool override {
    if (m_h_C.size() != m_h_C_reference.size()) {
      return false;
    }
    for (size_t i = 0; i < m_h_C.size(); i++) {
      if (std::abs(m_h_C[i] - m_h_C_reference[i]) > 1e-3f) {
        return false;
      } // NOLINT
    }
    return true;
  }

protected:
  void register_options() override {
    Baseliner::IWorkload<BackendT>::register_options();
    this->add_option("MatrixMulWorkload", "wA", "Width of Matrix A", m_wA_base);
    this->add_option("MatrixMulWorkload", "hA", "Height of Matrix A", m_hA_base);
    this->add_option("MatrixMulWorkload", "wB", "Width of Matrix B", m_wB_base);
    this->add_option("MatrixMulWorkload", "block_size", "Block size (16 or 32)", m_block_size);
  }

private:
  // Scales dimensions using work_size — hA is scaled to increase workload
  void compute_dimensions() {
    m_wA = m_wA_base;
    m_hA = m_hA_base * static_cast<int>(this->get_work_size());
    m_hB = m_wA; // inner dimensions must match
    m_wB = m_wB_base;
    m_size_A = m_wA * m_hA;
    m_size_B = m_wB * m_hB;
  }

  void allocate_host_buffers() {
    m_h_A.resize(m_size_A);
    m_h_B.resize(m_size_B);
    m_h_C.resize(m_hA * m_wB);
  }

  // Base dimensions (1 work_size unit = 32MFLOPs)
  int m_wA_base = 256; // NOLINT
  int m_hA_base = 256; // NOLINT
  int m_wB_base = 256; // NOLINT
  int m_block_size = 16;

  // Scaled dimensions
  int m_wA = 0, m_hA = 0;
  int m_wB = 0, m_hB = 0;
  int m_size_A = 0, m_size_B = 0;

  // Host buffers
  std::vector<float> m_h_A;
  std::vector<float> m_h_B;
  std::vector<float> m_h_C;
  std::vector<float> m_h_C_reference;

  // Device pointers
  float *m_d_A = nullptr;
  float *m_d_B = nullptr;
  float *m_d_C = nullptr;

  // Kernel launch config
  int m_threads_x;
  int m_threads_y;
  int m_grid_x;
  int m_grid_y;
};

#endif // MATMUL_HPP