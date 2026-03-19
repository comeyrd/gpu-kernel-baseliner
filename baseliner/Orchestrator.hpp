#ifndef BASELINER_ORCHESTRATOR_HPP
#define BASELINER_ORCHESTRATOR_HPP
#include "baseliner/Output.hpp"
#include "baseliner/Version.hpp"
#include <baseliner/Builder.hpp>
#include <baseliner/GIT_VERSION.hpp>
#include <baseliner/Planner.hpp>
#include <baseliner/Protocol.hpp>
#include <vector>
namespace Baseliner {
  namespace Orchestrator {

    inline auto run_plan(const Plan &plan, StorageManager *storage_manager = StorageManager::instance()) -> RunReport {
      IBenchmarkFactory bench_factory = Builder::build(plan, storage_manager);
      BenchmarkReport bench_report = bench_factory()->run_benchmark();
      return {plan, bench_report};
    };

    inline auto run_protocol(const Protocol &protocol) -> Report {
      Report report;
      auto *storage_manager = StorageManager::instance();
      report.m_baseliner_version = Version::string();
      report.m_git_version = BASELINER_GIT_VERSION;
      report.m_datetime = "";
      std::vector<Plan> plans = Planner::plan(protocol, storage_manager);
      for (const auto &plan : plans) {
        report.m_runs.push_back(run_plan(plan));
      }
      return report;
    };

    inline auto run_protocols(const std::vector<Protocol> &protocols) -> std::vector<Report> {
      std::vector<Report> reports;
      reports.reserve(protocols.size());
      for (const auto &protocol : protocols) {
        reports.push_back(run_protocol(protocol));
      }
      return reports;
    };
  }; // namespace Orchestrator

} // namespace Baseliner
#endif // BASELINER_ORCHESTRATOR_HPP