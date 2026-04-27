#ifndef BASELINER_SPECS_CUDABACKEND_HPP
#define BASELINER_SPECS_CUDABACKEND_HPP

#ifdef BASELINER_FULL_LIBRARY
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>
#else
namespace Baseliner::Hardware {
  using CudaBackend = Backend<cudaStream_t, std::monostate>;
} // namespace Baseliner::Hardware
#endif

#endif