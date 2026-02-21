#include "hip/hip_runtime.h"
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

#include <baseliner/backend/hip/HipBackend.hpp>
#include <cstdio>
#include <cstdlib>
namespace {
  __global__ void block_stream(const volatile int32_t *flag, volatile int32_t *timeout_flag, double timeout) {
    const auto start_clock = clock64();
    const bool use_timeout = timeout >= 0.;
    const long long timeout_cycles = use_timeout ? static_cast<long long>(timeout * 1e9) : 0;

    auto current_clock = clock64();
    while (!(*flag) && (!use_timeout || (current_clock - start_clock) < timeout_cycles)) {
      current_clock = clock64();
    }

    if (use_timeout && (current_clock - start_clock) >= timeout_cycles) {
      *timeout_flag = 1;
      __threadfence_system(); // Ensure timeout flag visibility on host.
      printf("\n Deadlock detected\n");
    }
  }
} // namespace

namespace Baseliner::Backend {
  template <>
  BlockingKernel<HipBackend>::BlockingKernel() {
    m_host_flag = new int;         // NOLINT
    m_host_timeout_flag = new int; // NOLINT
    CHECK_HIP(hipHostRegister(m_host_flag, sizeof(int), hipHostRegisterMapped));
    CHECK_HIP(hipHostGetDevicePointer((void **)&m_device_flag, m_host_flag, 0));
    CHECK_HIP(hipHostRegister(m_host_timeout_flag, sizeof(int), hipHostRegisterMapped));
    CHECK_HIP(hipHostGetDevicePointer((void **)&m_device_timeout_flag, m_host_timeout_flag, 0));
  }
  template <>
  BlockingKernel<HipBackend>::~BlockingKernel() {
    CHECK_HIP(hipGetLastError());
    if (m_host_flag != nullptr) {
      CHECK_HIP(hipHostUnregister(m_host_flag));
    }
    if (m_host_timeout_flag != nullptr) {
      CHECK_HIP(hipHostUnregister(m_host_timeout_flag));
    }
    delete m_host_flag;
    delete m_host_timeout_flag;
  }
  template <>
  void BlockingKernel<HipBackend>::block(std::shared_ptr<hipStream_t> stream, double timeout) {
    *m_host_flag = 0;
    *m_host_timeout_flag = 0;
    block_stream<<<1, 1, 0, *stream>>>(m_device_flag, m_device_timeout_flag, timeout);
  }

  template <>
  auto BlockingKernel<HipBackend>::operator=(BlockingKernel<HipBackend> &&other) noexcept
      -> BlockingKernel<HipBackend> & {
    {
      if (this != &other) {
        if (m_host_flag != nullptr) {
          CHECK_HIP(hipHostUnregister(m_host_flag));
        }
        if (m_host_timeout_flag != nullptr) {
          CHECK_HIP(hipHostUnregister(m_host_timeout_flag));
        }
        delete m_host_flag;
        delete m_host_timeout_flag;
        m_host_flag = other.m_host_flag;
        m_host_timeout_flag = other.m_host_timeout_flag;
        m_device_flag = other.m_device_flag;
        m_device_timeout_flag = other.m_device_timeout_flag;
        other.m_device_flag = nullptr;
        other.m_device_timeout_flag = nullptr;
        other.m_host_flag = nullptr;
        other.m_host_timeout_flag = nullptr;
      }
      return *this;
    }
  }
} // namespace Baseliner::Backend
