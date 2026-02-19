#ifndef GEMM_ROCBLAS_CASE_HPP
#define GEMM_ROCBLAS_CASE_HPP
#include "hip/hip_runtime.h"
#include <baseliner/Case.hpp>
#include <baseliner/backend/hip/HipBackend.hpp>
#include <random>
#include <rocblas/rocblas.h>
#include <type_traits>

namespace Detail {
  // Primary template
  template <typename T>
  struct RocBlasTraits;

  template <>
  struct RocBlasTraits<float> {
    static constexpr auto gemm = rocblas_sgemm;
    static constexpr auto gemm_64 = rocblas_sgemm_64;
    using Type = float;
  };

  template <>
  struct RocBlasTraits<double> {
    static constexpr auto gemm = rocblas_dgemm;
    static constexpr auto gemm_64 = rocblas_dgemm_64;
    using Type = double;
  };

  template <>
  struct RocBlasTraits<rocblas_half> {
    static constexpr auto gemm = rocblas_hgemm;
    static constexpr auto gemm_64 = rocblas_hgemm_64;
    using Type = rocblas_half;
  };

  template <>
  struct RocBlasTraits<rocblas_float_complex> {
    static constexpr auto gemm = rocblas_cgemm;
    static constexpr auto gemm_64 = rocblas_cgemm_64;
    using Type = rocblas_float_complex;
  };

  template <>
  struct RocBlasTraits<rocblas_double_complex> {
    static constexpr auto gemm = rocblas_zgemm;
    static constexpr auto gemm_64 = rocblas_zgemm_64;
    using Type = rocblas_double_complex;
  };
} // namespace Detail

template <typename T>
class RocBlasGemmCase : public Baseliner::ICase<Baseliner::Device::HipBackend> {
public:
  auto name() -> std::string override {
    return "RocBlasGemm_" + std::string(typeid(T).name());
  }

  void setup(std::shared_ptr<typename Baseliner::Device::HipBackend::stream_t> stream) override {
    rocblas_create_handle(&m_handle);
    rocblas_set_stream(m_handle, *stream);
    m_m = m_m_def * m_work_size;
    m_k = m_k_def * m_work_size;
    m_n = m_n_def * m_work_size;
    size_t size_A = m_m * m_k;
    size_t size_B = m_k * m_n;
    size_t size_C = m_m * m_n;

    std::vector<T> h_A(size_A);
    std::vector<T> h_B(size_B);

    generate_random_data(h_A);
    generate_random_data(h_B);

    CHECK_HIP(hipMalloc(&m_d_A, size_A * sizeof(T)));
    CHECK_HIP(hipMalloc(&m_d_B, size_B * sizeof(T)));
    CHECK_HIP(hipMalloc(&m_d_C, size_C * sizeof(T)));

    CHECK_HIP(hipMemcpyAsync(m_d_A, h_A.data(), size_A * sizeof(T), hipMemcpyHostToDevice, *stream));
    CHECK_HIP(hipMemcpyAsync(m_d_B, h_B.data(), size_B * sizeof(T), hipMemcpyHostToDevice, *stream));
  }

  void run_case(std::shared_ptr<typename Baseliner::Device::HipBackend::stream_t> stream) override {
    Detail::RocBlasTraits<T>::gemm(m_handle, rocblas_operation_none, rocblas_operation_none, m_m, m_n, m_k, &m_alpha,
                                   m_d_A, m_m, m_d_B, m_k, &m_beta, m_d_C, m_m);
  }

  void setup_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) override {
    uint64_t M = static_cast<uint64_t>(m_m);
    uint64_t N = static_cast<uint64_t>(m_n);
    uint64_t K = static_cast<uint64_t>(m_k);
    uint64_t total_flops = 2ULL * M * N * K;
    engine->register_metric<Baseliner::Stats::FLOPCount>(total_flops);
  }

  void update_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) override {
    uint64_t M = static_cast<uint64_t>(m_m);
    uint64_t N = static_cast<uint64_t>(m_n);
    uint64_t K = static_cast<uint64_t>(m_k);
    uint64_t total_flops = 2ULL * M * N * K;
    engine->update_values<Baseliner::Stats::FLOPCount>(total_flops);
  }
  void teardown(std::shared_ptr<typename Baseliner::Device::HipBackend::stream_t> stream) override {
    rocblas_destroy_handle(m_handle);
    CHECK_HIP(hipFree(m_d_A));
    CHECK_HIP(hipFree(m_d_B));
    CHECK_HIP(hipFree(m_d_C));
  }

  void reset_case(std::shared_ptr<typename Baseliner::Device::HipBackend::stream_t> stream) override {
    CHECK_HIP(hipMemsetAsync(m_d_C, 0, (m_m * m_n) * sizeof(T), *stream));
  }

  auto validate_case() -> bool override {
    return true;
  }
  void register_options() override {
    add_option("RocBlasGemm", "work_size", "The work size", m_work_size);
    add_option("RocBlasGemm", "seed", "The seed", m_seed);
  }

private:
  void generate_random_data(std::vector<T> &vec) {
    std::mt19937 gen(m_seed);

    if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dist(0.0, 1.0);
      for (auto &val : vec)
        val = dist(gen);
    } else {
      // Fallback for custom types like rocblas_half
      for (auto &val : vec)
        val = static_cast<T>(static_cast<float>(gen()) / RAND_MAX);
    }
  }

  rocblas_handle m_handle;
  T *m_d_A = nullptr, *m_d_B = nullptr, *m_d_C = nullptr;
  int m_m_def = 1024, m_n_def = 1024, m_k_def = 1024;
  int m_m, m_n, m_k;
  int m_work_size = 1;
  T m_alpha = 1.0, m_beta = 0.0;
  int m_seed = 42;
};
#endif // GEMM_ROCBLAS_CASE_HPP
