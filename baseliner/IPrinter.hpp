#ifndef BASELINER_IPRINTER_HPP
#define BASELINER_IPRINTER_HPP
#include "baseliner/Output.hpp"
namespace Baseliner {
  class IBenchmarkPrinter {
  public:
    virtual void consume_single_run_report(const SingleRunReport &report) = 0;
    virtual ~IBenchmarkPrinter() = default;
  };
} // namespace Baseliner
#endif // BASELINER_IPRINTER_HPP