#ifndef BASELINER_SPECS_CUDABACKEND_HPP
#define BASELINER_SPECS_CUDABACKEND_HPP

#ifdef BASELINER_FULL_LIBRARY
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>
#else
namespace Baseliner::Hardware {
  using CudaBackend = Backend<cudaStream_t, std::monostate>;
} // namespace Baseliner::Hardware

inline void check_cuda_error(cudaError_t error_code, const char *file, int line) {
  if (error_code != cudaSuccess) {
    throw Baseliner::Errors::hardware_error("CUDA", cudaGetErrorString(error_code), file, line);
  }
}
inline void check_cuda_error_no_except(cudaError_t error_code, const char *file, int line) {
  if (error_code != cudaSuccess) {
    auto msg = Baseliner::Errors::hardware_error_noexcept("CUDA", cudaGetErrorString(error_code), file, line);
    std::cerr << msg << std::endl;
  }
}
#define CHECK_CUDA(error) check_cuda_error(error, __FILE__, __LINE__)                     // NOLINT
#define CHECK_CUDA_NO_EXCEPT(error) check_cuda_error_no_except(error, __FILE__, __LINE__) // NOLINT

#endif

#endif