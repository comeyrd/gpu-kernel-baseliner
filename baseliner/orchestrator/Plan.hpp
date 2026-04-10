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
    std::string impl;
    std::string preset;
    OptionsMap options;
  };
  DESCRIBE(PlannedComponent, FIELD(impl), FIELD(preset), FIELD(options))

  struct PlannedStat {
    std::string preset;
    std::vector<std::string> stats;
    OptionsMap options;
  };
  DESCRIBE(PlannedStat, FIELD(preset), FIELD(stats), FIELD(options))

  struct BenchmarkPlan {
    PlannedComponent workload;
    PlannedComponent backend;
    PlannedComponent benchmark;
    PlannedComponent stopping;
    PlannedStat stats;
    std::optional<SweepSpec> sweep;
  };
  DESCRIBE(BenchmarkPlan, FIELD(workload), FIELD(backend), FIELD(benchmark), FIELD(stopping), FIELD(stats),
           FIELD(sweep))

  struct CampaignPlan {
    std::string name;
    std::string recipe_name;
    Recipe recipe;
    std::vector<BenchmarkPlan> benchmarks;
    OnIncompatible on_incompatible;
  };
  DESCRIBE(CampaignPlan, FIELD(name), FIELD(recipe_name), FIELD(recipe), FIELD(benchmarks), FIELD(on_incompatible))

} // namespace Baseliner
#endif // BASELINER_OUTPUT_HPP