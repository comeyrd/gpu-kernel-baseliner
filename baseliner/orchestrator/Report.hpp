#ifndef BASELINER_ORCHESTRATOR_REPORT_HPP
#define BASELINER_ORCHESTRATOR_REPORT_HPP
#include <baseliner/orchestrator/Plan.hpp>
#include <string>
#include <vector>
namespace Baseliner {
  struct RunReport {
    BenchmarkPlan plan;
    BenchmarkReport benchmark_report;
  };
  DESCRIBE(RunReport, FIELD(plan), FIELD(benchmark_report))

  struct CampaignReport {
    std::string name;
    std::string recipe_name;
    Recipe recipe;
    // Key1 Backend Key2 Case
    std::unordered_map<std::string, std::unordered_map<std::string, RunReport>> benchmark_runs;
  };
  DESCRIBE(CampaignReport, FIELD(name), FIELD(recipe_name), FIELD(recipe), FIELD(benchmark_runs))

  inline auto campaign_plan_from_report(const CampaignReport &report,
                                        OnIncompatible on_incompatible = OnIncompatible::Skip) -> CampaignPlan {
    CampaignPlan plan;
    plan.name = report.name;
    plan.recipe_name = report.recipe_name;
    plan.recipe = report.recipe;
    plan.on_incompatible = on_incompatible;

    for (const auto &[backend, workloads_map] : report.benchmark_runs) {
      for (const auto &[workload_name, run_report] : workloads_map) {
        plan.benchmarks.push_back(run_report.plan);
      }
    }

    return plan;
  }

  struct Report {
    std::string baseliner_version;
    std::string git_version;
    std::string datetime;
    std::vector<CampaignReport> campaign_runs;
  };
  DESCRIBE(Report, FIELD(baseliner_version), FIELD(git_version), FIELD(datetime), FIELD(campaign_runs))

} // namespace Baseliner

#endif