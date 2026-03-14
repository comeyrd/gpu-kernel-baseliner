#include "amd_smi/amdsmi.h"
#include <baseliner/Error.hpp>
#include <baseliner/hardware/hip/HipBackend.hpp>
#include <iostream>
#include <string>
void check_amd_smi_error(amdsmi_status_t error_code, const char *file, int line) {
  if (error_code != AMDSMI_STATUS_SUCCESS) {
    const char *str_code;
    amdsmi_status_code_to_string(error_code, &str_code);
    throw Baseliner::Errors::hardware_error("AmdSmi", str_code, file, line);
  }
}

void check_amd_smi_error_no_except(amdsmi_status_t error_code, const char *file, int line) {
  if (error_code != AMDSMI_STATUS_SUCCESS) {
    const char *str_code;
    amdsmi_status_code_to_string(error_code, &str_code);
    auto msg = Baseliner::Errors::hardware_error_noexcept("AmdSmi", str_code, file, line);
    std::cerr << msg << std::endl;
  }

} // namespace Baseliner