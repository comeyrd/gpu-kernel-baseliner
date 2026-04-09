#ifndef BASELINER_CORE_BENCHMARKREPORT_HPP
#define BASELINER_CORE_BENCHMARKREPORT_HPP
#include <baseliner/core/Metric.hpp>
#include <baseliner/core/Options.hpp>
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

} // namespace Baseliner
#endif // BASELINER_CORE_BENCHMARKREPORT_HPP