/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

// Copyright 2026, Come Eyraud.

#include <baseliner/backend/cuda/CudaBackend.hpp>
#include <cstdio>
#include <cstdlib>
namespace {
  __global__ void block_stream(const volatile int32_t *flag, volatile int32_t *timeout_flag, double timeout) {
    const auto start_clock = clock64();
    const bool use_timeout = timeout >= 0.;
    const long long timeout_cycles = use_timeout ? static_cast<long long>(timeout * 1e9) : 0;

    auto current_clock = clock64();
    while (((*flag) == 0) && (!use_timeout || (current_clock - start_clock) < timeout_cycles)) {
      current_clock = clock64();
    }

    if (use_timeout && (current_clock - start_clock) >= timeout_cycles) {
      *timeout_flag = 1;
      __threadfence_system();           // Ensure timeout flag visibility on host.
      printf("\n Deadlock detected\n"); // NOLINT
    }
  }
} // namespace

namespace Baseliner::Backend {
  template <>
  void BlockingKernel<CudaBackend>::alloc(int device) {
    m_host_flag_v[device] = new int;         // NOLINT
    m_host_timeout_flag_v[device] = new int; // NOLINT
    CHECK_CUDA(cudaHostRegister(m_host_flag_v[device], sizeof(int), cudaHostRegisterMapped));
    CHECK_CUDA(cudaHostGetDevicePointer(&m_device_flag_v[device], m_host_flag_v[device], 0));
    CHECK_CUDA(cudaHostRegister(m_host_timeout_flag_v[device], sizeof(int), cudaHostRegisterMapped));
    CHECK_CUDA(cudaHostGetDevicePointer(&m_device_timeout_flag_v[device], m_host_timeout_flag_v[device], 0));
  }
  template <>
  void BlockingKernel<CudaBackend>::free(int device) {
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaHostUnregister(m_host_flag_v[device]));
    CHECK_CUDA(cudaHostUnregister(m_host_timeout_flag_v[device]));
    delete m_host_flag_v[device];
    delete m_host_timeout_flag_v[device];
  }
  template <>
  void BlockingKernel<CudaBackend>::block(std::shared_ptr<cudaStream_t> stream, double timeout) {
    int current_device = CudaBackend::instance()->get_current_device();
    *m_host_flag_v[current_device] = 0;
    *m_host_timeout_flag_v[current_device] = 0;
    block_stream<<<1, 1, 0, *stream>>>(m_device_flag_v[current_device], m_device_timeout_flag_v[current_device],
                                       timeout);
  }
} // namespace Baseliner::Backend
