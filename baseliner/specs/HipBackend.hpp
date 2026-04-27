#ifndef BASELINER_SPECS_HIPBACKEND_HPP
#define BASELINER_SPECS_HIPBACKEND_HPP

#ifdef BASELINER_FULL_LIBRARY
#include <baseliner/core/hardware/hip/HipBackend.hpp>
#else
namespace Baseliner::Hardware {
  using HipBackend = Backend<hipStream_t, std::monostate>;
} // namespace Baseliner::Hardware
inline void check_hip_error(hipError_t error_code, const char *file, int line) {
  if (error_code != hipSuccess) {
    throw Baseliner::Errors::hardware_error("HIP", hipGetErrorString(error_code), file, line);
  }
}
inline void check_hip_error_no_except(hipError_t error_code, const char *file, int line) {
  if (error_code != hipSuccess) {
    auto msg = Baseliner::Errors::hardware_error_noexcept("HIP", hipGetErrorString(error_code), file, line);
    std::cerr << msg << std::endl;
  }
}
#define CHECK_HIP(error) check_hip_error(error, __FILE__, __LINE__)                     // NOLINT
#define CHECK_HIP_NO_EXCEPT(error) check_hip_error_no_except(error, __FILE__, __LINE__) // NOLINT

#endif

#endif