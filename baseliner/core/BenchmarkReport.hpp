#ifndef BASELINER_CORE_BENCHMARKREPORT_HPP
#define BASELINER_CORE_BENCHMARKREPORT_HPP
#include <baseliner/core/Metric.hpp>
#include <baseliner/core/Options.hpp>

#include <baseliner/core/hardware/Backend.hpp>
#include <baseliner/utils/Utils.hpp>
#include <optional>
#include <vector>
namespace Baseliner {

  struct RunReport {
    std::string id = Utils::gen_uuid();
    std::optional<OptionsMap> sweep_point; // Interface → option → value
    std::vector<Metric> measurements;
  };
  DESCRIBE(RunReport, FIELD(id), FIELD(sweep_point), FIELD(measurements))
  struct BenchmarkReport {
    std::string id = Utils::gen_uuid();
    std::vector<RunReport> results;
    Hardware::HardwareInfo hardware;
  };
  DESCRIBE(BenchmarkReport, FIELD(id), FIELD(results), FIELD(hardware))

} // namespace Baseliner
#endif // BASELINER_CORE_BENCHMARKREPORT_HPP