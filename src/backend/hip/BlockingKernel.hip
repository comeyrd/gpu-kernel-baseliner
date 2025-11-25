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
#include "HipBackend.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>
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
namespace Baseliner {
  namespace Backend {
    HipBackend::BlockingKernel::BlockingKernel() {
      CHECK_HIP(hipHostRegister(&m_host_flag, sizeof(m_host_flag), hipHostRegisterMapped));
      CHECK_HIP(hipHostGetDevicePointer((void **)&m_device_flag, &m_host_flag, 0));
      CHECK_HIP(hipHostRegister(&m_host_timeout_flag, sizeof(m_host_timeout_flag), hipHostRegisterMapped));
      CHECK_HIP(hipHostGetDevicePointer((void **)&m_device_timeout_flag, &m_host_timeout_flag, 0));
    }
    HipBackend::BlockingKernel::~BlockingKernel() {
      CHECK_HIP(hipGetLastError());
      CHECK_HIP(hipHostUnregister(&m_host_flag));
      CHECK_HIP(hipHostUnregister(&m_host_timeout_flag));
    }
    void HipBackend::BlockingKernel::block(hipStream_t stream, double timeout) {
      m_host_flag = 0;
      m_host_timeout_flag = 0;
      block_stream<<<1, 1, 0, stream>>>(m_device_flag, m_device_timeout_flag, timeout);
    }
  } // namespace Backend

} // namespace Baseliner
