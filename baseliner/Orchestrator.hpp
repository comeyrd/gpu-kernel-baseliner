#ifndef BASELINER_ORCHESTRATOR_HPP
#define BASELINER_ORCHESTRATOR_HPP
#include "baseliner/Output.hpp"
#include "baseliner/Version.hpp"
#include <baseliner/Builder.hpp>
#include <baseliner/GIT_VERSION.hpp>
#include <baseliner/Planner.hpp>
#include <baseliner/Protocol.hpp>
#include <baseliner/RQ.hpp>
#include <vector>
namespace Baseliner {
  namespace Orchestrator {
    inline void load_presets(const Protocol &preset_protocol) {
      StorageManager::instance()->load_protocol_presets(preset_protocol);
    };

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
        if (ExecutionController::exit_requested()) {
          break;
        }
        report.m_runs.push_back(run_plan(plan));
      }
      return report;
    };

    inline auto replay_runs(const Report &to_replay_report) -> Report {
      Report report;
      report.m_baseliner_version = Version::string();
      report.m_git_version = BASELINER_GIT_VERSION;
      report.m_datetime = "";
      for (const RunReport &run : to_replay_report.m_runs) {
        report.m_runs.push_back(run_plan(run.m_plan));
      }
      return report;
    };
    inline auto run_research_questions(const std::vector<std::string> &case_names) -> Report {
      std::vector<std::string> backends = StorageManager::instance()->list_backends();
      std::vector<RecipeComponent> case_components;
      case_components.reserve(case_names.size());
      std::vector<RecipeComponent> backends_components;
      backends_components.reserve(backends.size());

      for (const auto &case_name : case_names) {
        case_components.push_back({case_name, {}});
      }
      for (const auto &backend : backends) {
        backends_components.push_back({backend, {}});
      }
      Protocol protocol = rq_protocol(RQSize::Small, case_components, backends_components);
      return run_protocol(protocol);
    };

    inline auto run_protocols(const std::vector<Protocol> &protocols) -> std::vector<Report> {
      std::vector<Report> reports;
      reports.reserve(protocols.size());
      for (const auto &protocol : protocols) {
        if (ExecutionController::exit_requested()) {
          break;
        }
        reports.push_back(run_protocol(protocol));
      }
      return reports;
    };
  }; // namespace Orchestrator

} // namespace Baseliner
#endif // BASELINER_ORCHESTRATOR_HPP