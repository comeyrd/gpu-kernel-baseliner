#include "GemmRocblasCase.hpp"
#include <baseliner/Axe.hpp>
#include <baseliner/Benchmark.hpp>
#include <baseliner/Suite.hpp>
#include <baseliner/backend/BackendStats.hpp>
#include <baseliner/backend/hip/HipBackend.hpp>
#include <hip/hip_runtime.h>
#include <iostream>
#include <rocblas/rocblas.h>
#include <vector>

using RocBlasGemm_F16 = RocBlasGemmCase<rocblas_half>;
using RocBlasGemm_F32 = RocBlasGemmCase<float>;
using RocBlasGemm_F64 = RocBlasGemmCase<double>;

using RocBlasGemm_C32 = RocBlasGemmCase<rocblas_float_complex>;
using RocBlasGemm_C64 = RocBlasGemmCase<rocblas_double_complex>;

namespace ase {

  auto benchmark_f16 = Baseliner::HipBenchmark()
                           .set_case<RocBlasGemm_F32>()
                           .add_stat<Baseliner::Stats::Q1>()
                           .add_stat<Baseliner::Stats::Q3>()
                           .add_stat<Baseliner::Stats::Median>()
                           .add_stat<Baseliner::Stats::WithoutOutliers>()
                           .add_stat<Baseliner::Stats::MedianAbsoluteDeviation>()
                           .add_stat<Baseliner::Stats::ClockFrequency<Baseliner::Backend::HipBackend>>()
                           .add_stat<Baseliner::Stats::FLOPThroughput>();

  Baseliner::Axe axe = {"RocBlasGemm", "work_size", {"1", "2", "3", "4", "5", "6", "7", "8"}};

  auto bench_suite = std::make_shared<Baseliner::SingleAxeSuite>(
      std::make_shared<Baseliner::HipBenchmark>(std::move(benchmark_f16)), std::move(axe));

  BASELINER_REGISTER_TASK(bench_suite.get());
} // namespace ase