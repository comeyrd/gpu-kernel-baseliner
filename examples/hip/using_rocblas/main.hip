#include "GemmRocblasCase.hpp"
#include <baseliner/Axe.hpp>
#include <baseliner/Benchmark.hpp>
#include <baseliner/Suite.hpp>
#include <baseliner/hardware/BackendStats.hpp>
#include <baseliner/hardware/hip/HipBackend.hpp>
#include <baseliner/managers/RegisteringMacros.hpp>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

using RocBlasGemm_F16 = RocBlasGemmCase<rocblas_half>;
using RocBlasGemm_F32 = RocBlasGemmCase<float>;
using RocBlasGemm_F64 = RocBlasGemmCase<double>;

using RocBlasGemm_C32 = RocBlasGemmCase<rocblas_float_complex>;
using RocBlasGemm_C64 = RocBlasGemmCase<rocblas_double_complex>;

BASELINER_REGISTER_KERNEL(RocBlasGemm_F16);
BASELINER_REGISTER_KERNEL(RocBlasGemm_F32);
BASELINER_REGISTER_KERNEL(RocBlasGemm_F64);
BASELINER_REGISTER_KERNEL(RocBlasGemm_C32);
BASELINER_REGISTER_KERNEL(RocBlasGemm_C64);
