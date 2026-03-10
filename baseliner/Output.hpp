#ifndef BASELINER_OUTPUT_HPP
#define BASELINER_OUTPUT_HPP
#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Protocol.hpp>
#include <baseliner/hardware/Backend.hpp>
#include <string>
#include <vector>
namespace Baseliner {

  struct BenchmarkResult {
    OptionsMap m_sweep_point; // Interface → option → value
    std::vector<Metric> m_measurements;
  };
  struct PlannedComponent {
    std::string m_impl;
    std::string m_preset;
    OptionsMap m_options;
  };
  struct PlannedSweep {
    std::string m_interface;
    std::string m_option;
    std::vector<std::string> m_values;
  };

  struct Plan {
    std::string m_campaign_name;
    std::string m_recipe_name;
    PlannedComponent m_case;
    PlannedComponent m_backend;
    PlannedComponent m_benchmark;
    PlannedComponent m_stopping;
    PlannedComponent m_stats;
    std::optional<PlannedSweep> m_sweep;
  };

  struct RunReport {
    Plan m_plan;
    std::vector<BenchmarkResult> m_results;
    Hardware::HardwareInfo m_hardware;
  };

  struct Report {
    std::string m_baseliner_version;
    std::string m_git_version;
    std::string m_datetime;
    std::vector<RunReport> m_runs;
  };
} // namespace Baseliner
#endif // BASELINER_OUTPUT_HPP