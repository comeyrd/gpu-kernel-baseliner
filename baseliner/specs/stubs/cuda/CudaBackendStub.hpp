#ifndef BASELINER_SPECS_STUBS_CUDA_CUDABACKENDSTUB_HPP
#define BASELINER_SPECS_STUBS_CUDA_CUDABACKENDSTUB_HPP
#include "cuda_runtime.h"
#include <baseliner/specs/Options.hpp>

#include <baseliner/specs/stubs/BackendStub.hpp>

namespace Baseliner::Hardware {
  using CudaBackend = Backend<cudaStream_t, std::monostate>;
} // namespace Baseliner::Hardware
#endif // BASELINER_SPECS_STUBS_BACKENDSTUB_HPP
