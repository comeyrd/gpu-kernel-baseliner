#include <baseliner/Error.hpp>
#include <baseliner/hardware/cuda/CudaBackend.hpp>
#include <iostream>
#include <nvml.h>
#include <string>

void check_nvml_error(nvmlReturn_t error_code, const char *file, int line) {
  if (error_code != NVML_SUCCESS) {
    throw Baseliner::Errors::hardware_error("NVML", nvmlErrorString(error_code),
                                            file, line);
    ;
  }
}

void check_nvml_error_no_except(nvmlReturn_t error_code, const char *file,
                                int line) {
  if (error_code != NVML_SUCCESS) {
    auto msg = Baseliner::Errors::hardware_error_noexcept(
        "NVML", nvmlErrorString(error_code), file, line);
    std::cerr << msg << std::endl;
  }
}
