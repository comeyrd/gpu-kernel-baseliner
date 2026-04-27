#ifndef BASELINER_SPECS_STUBS_HIP_HIPBACKENDSTUB_HPP
#define BASELINER_SPECS_STUBS_HIP_HIPBACKENDSTUB_HPP
#include "hip/hip_runtime.h"
#include <baseliner/specs/Options.hpp>

#include <baseliner/specs/stubs/BackendStub.hpp>
namespace Baseliner::Hardware {
  using HipBackend = Backend<hipStream_t, std::monostate>;
} // namespace Baseliner::Hardware
#endif // BASELINER_SPECS_STUBS_BACKENDSTUB_HPP
