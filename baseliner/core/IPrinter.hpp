#ifndef BASELINER_CORE_IPRINTER_HPP
#define BASELINER_CORE_IPRINTER_HPP
#include <baseliner/core/BenchmarkReport.hpp>

namespace Baseliner {
  class IBenchmarkPrinter {
  public:
    virtual void consume_single_run_report(const SingleRunReport &report) = 0;
    virtual ~IBenchmarkPrinter() = default;
  };
} // namespace Baseliner
#endif // BASELINER_IPRINTER_HPP