#ifndef BASELINER_OUTPUT_HPP
#define BASELINER_OUTPUT_HPP
#include <baseliner/Metric.hpp>
#include <baseliner/Protocol.hpp>
#include <baseliner/hardware/Backend.hpp>
#include <string>
#include <vector>
namespace Baseliner {

  struct PlannedComponent {
    std::string m_impl;
    std::string m_preset;
    OptionsMap m_options;
  };
  struct PlannedStat {
    std::string m_preset;
    std::vector<std::string> m_stats;
    OptionsMap m_options;
  };

  struct Plan {
    std::string m_campaign_name;
    std::string m_recipe_name;
    PlannedComponent m_case;
    PlannedComponent m_backend;
    PlannedComponent m_benchmark;
    PlannedComponent m_stopping;
    PlannedStat m_stats;
    std::optional<SweepSpec> m_sweep;
    OnIncompatible m_on_incompatible;
  };
  struct SingleRunReport {
    std::optional<OptionsMap> m_sweep_point; // Interface → option → value
    std::vector<Metric> m_measurements;
  };
  struct BenchmarkReport {
    std::vector<SingleRunReport> m_results;
    Hardware::HardwareInfo m_hardware;
  };
  struct RunReport {
    Plan m_plan;
    BenchmarkReport m_benchmark_report;
  };
  struct Report {
    std::string m_baseliner_version;
    std::string m_git_version;
    std::string m_datetime;
    std::vector<RunReport> m_runs;
  };
} // namespace Baseliner
#endif // BASELINER_OUTPUT_HPP