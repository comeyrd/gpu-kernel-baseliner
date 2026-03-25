#ifndef BASELINER_IPRINTER_HPP
#define BASELINER_IPRINTER_HPP
#include <baseliner/core/Metric.hpp>
#include <baseliner/core/hardware/Backend.hpp>
#include <optional>
#include <vector>
namespace Baseliner {
  struct SingleRunReport {
    std::optional<OptionsMap> m_sweep_point; // Interface → option → value
    std::vector<Metric> m_measurements;
  };
  struct BenchmarkReport {
    std::vector<SingleRunReport> m_results;
    Hardware::HardwareInfo m_hardware;
  };

  class IBenchmarkPrinter {
  public:
    virtual void consume_single_run_report(const SingleRunReport &report) = 0;
    virtual ~IBenchmarkPrinter() = default;
  };
} // namespace Baseliner
#endif // BASELINER_IPRINTER_HPP