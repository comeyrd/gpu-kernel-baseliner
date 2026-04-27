#ifndef BASELINER_SPECS_STUBS_BACKENDSTUB_HPP
#define BASELINER_SPECS_STUBS_BACKENDSTUB_HPP
#include <baseliner/specs/Options.hpp>

namespace Baseliner::Hardware {

  template <typename S, typename O>
  class Backend : public IOption {
  public:
    using stream_t = S;
    using launch_result_t = O;
  };
} // namespace Baseliner::Hardware
#endif // BASELINER_SPECS_STUBS_BACKENDSTUB_HPP
