#include <baseliner/backend/hip/HipBackend.hpp>
#include <iostream>
#include "amd_smi/amdsmi.h"
#include <string>

void check_amd_smi_error(amdsmi_status_t error_code, const char *file, int line) {
  if (error_code != AMDSMI_STATUS_SUCCESS) {
    const char* str_code;
    amdsmi_status_code_to_string(error_code,& str_code);
    std::string msg = std::string("AmdSmi Error : ") + std::string(str_code) + std::string(" in : ") + file +
                      std::string(" line ") + std::to_string(line);
    throw std::runtime_error(msg);
  }
}

void check_amd_smi_error_no_except(amdsmi_status_t error_code, const char *file, int line) {
  if (error_code != AMDSMI_STATUS_SUCCESS) {
    const char* str_code;
    amdsmi_status_code_to_string(error_code,& str_code);
    std::string msg = std::string("AmdSmi Error : ") + std::string(str_code) + std::string(" in : ") + file +
                      std::string(" line ") + std::to_string(line);
    std::cerr << msg << std::endl;
  }
}