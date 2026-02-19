#include <baseliner/backend/cuda/CudaBackend.hpp>
#include <iostream>
#include <nvml.h>
#include <string>

void check_nvml_error(nvmlReturn_t error_code, const char *file, int line) {
  if (error_code != NVML_SUCCESS) {
    std::string msg = std::string("NVML Error : ") + nvmlErrorString(error_code) + std::string(" in : ") + file +
                      std::string(" line ") + std::to_string(line);
    throw std::runtime_error(msg);
  }
}

void check_nvml_error_no_except(nvmlReturn_t error_code, const char *file, int line) {
  if (error_code != NVML_SUCCESS) {
    std::string msg = std::string("NVML Error : ") + nvmlErrorString(error_code) + std::string(" in : ") + file +
                      std::string(" line ") + std::to_string(line);
    std::cerr << msg << std::endl;
  }
}