#ifndef BASELINER_ORCHESTRATOR_PLAN_HPP
#define BASELINER_ORCHESTRATOR_PLAN_HPP
#include <baseliner/core/Benchmark.hpp>
#include <baseliner/core/Metric.hpp>
#include <baseliner/core/hardware/Backend.hpp>
#include <baseliner/orchestrator/Protocol.hpp>
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
} // namespace Baseliner
#endif // BASELINER_OUTPUT_HPP