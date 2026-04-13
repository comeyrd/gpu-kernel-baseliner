#ifndef BASELINER_CORE_BENCHMARKREPORT_HPP
#define BASELINER_CORE_BENCHMARKREPORT_HPP
#include <baseliner/core/Metric.hpp>
#include <baseliner/core/Options.hpp>
#include <baseliner/core/hardware/Backend.hpp>
#include <optional>
#include <vector>
namespace Baseliner {

  struct SingleRunReport {
    std::optional<OptionsMap> sweep_point; // Interface → option → value
    std::vector<Metric> measurements;
  };
  DESCRIBE(SingleRunReport, FIELD(sweep_point), FIELD(measurements))
  struct BenchmarkReport {
    std::vector<SingleRunReport> results;
    Hardware::HardwareInfo hardware;
  };
  DESCRIBE(BenchmarkReport, FIELD(results), FIELD(hardware))

} // namespace Baseliner
#endif // BASELINER_CORE_BENCHMARKREPORT_HPP