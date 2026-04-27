#ifndef BASELINER_SPECS_HIPBACKEND_HPP
#define BASELINER_SPECS_HIPBACKEND_HPP

#ifdef BASELINER_FULL_LIBRARY
#include <baseliner/core/hardware/hip/HipBackend.hpp>
#else
namespace Baseliner::Hardware {
  using HipBackend = Backend<hipStream_t, std::monostate>;
} // namespace Baseliner::Hardware
#endif

#endif