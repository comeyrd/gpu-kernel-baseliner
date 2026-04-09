#ifndef BASELINER_ORCHESTRATOR_REPORT_HPP
#define BASELINER_ORCHESTRATOR_REPORT_HPP
#include <baseliner/orchestrator/Plan.hpp>
#include <string>
#include <vector>
namespace Baseliner {
  struct RunReport {
    BenchmarkPlan m_plan;
    BenchmarkReport m_benchmark_report;
  };

  struct CampaignReport {
    std::string name;
    std::string recipe_name;
    Recipe recipe;
    // Key1 Backend Key2 Case
    std::unordered_map<std::string, std::unordered_map<std::string, RunReport>> benchmark_runs;
  };

  inline auto campaign_plan_from_report(const CampaignReport &report,
                                        OnIncompatible on_incompatible = OnIncompatible::Skip) -> CampaignPlan {
    CampaignPlan plan;
    plan.name = report.name;
    plan.recipe_name = report.recipe_name;
    plan.recipe = report.recipe;
    plan.on_incompatible = on_incompatible;

    for (const auto &[backend, cases_map] : report.benchmark_runs) {
      for (const auto &[case_name, run_report] : cases_map) {
        plan.benchmarks.push_back(run_report.m_plan);
      }
    }

    return plan;
  }

  struct Report {
    std::string m_baseliner_version;
    std::string m_git_version;
    std::string m_datetime;
    std::vector<CampaignReport> m_campaign_runs;
  };
} // namespace Baseliner

#endif