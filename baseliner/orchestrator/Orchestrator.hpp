#ifndef BASELINER_ORCHESTRATOR_ORCHESTRATOR_HPP
#define BASELINER_ORCHESTRATOR_ORCHESTRATOR_HPP
#include <baseliner/RQ.hpp>
#include <baseliner/cli/CliHelper.hpp>
#include <baseliner/core/GIT_VERSION.hpp>
#include <baseliner/core/Version.hpp>
#include <baseliner/orchestrator/Builder.hpp>
#include <baseliner/orchestrator/Plan.hpp>
#include <baseliner/orchestrator/Planner.hpp>
#include <baseliner/orchestrator/Protocol.hpp>
#include <baseliner/orchestrator/Report.hpp>
#include <vector>
namespace Baseliner {
  namespace Orchestrator {
    inline void load_presets(const Protocol &preset_protocol) {
      StorageManager::instance()->load_protocol_presets(preset_protocol);
    };

    inline auto run_plan(const Plan &plan, StorageManager *storage_manager = StorageManager::instance()) -> RunReport {
      std::shared_ptr<Cli::CliPrinter> printer = std::make_shared<Cli::CliPrinter>();
      IBenchmarkFactory bench_factory = Builder::build(plan, storage_manager);
      printer->print_plan(plan);
      std::shared_ptr<IBenchmark> bench = bench_factory();
      bench->set_printer(printer);
      BenchmarkReport bench_report = bench->run_benchmark();
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
      Protocol protocol = rq_protocol(RQSize::Medium, case_components, backends_components);
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
    inline auto get_default_protocol() -> Protocol {
      Protocol protocol;
      auto *storage_manager = StorageManager::instance();
      protocol.m_baseliner_version = Version::string();
      protocol.m_presets = storage_manager->get_all_component_presets();
      protocol.m_stats_presets = storage_manager->get_all_stats_presets();
      Recipe def_recipe;
      def_recipe.m_stats = RecipeStat{"default"};
      def_recipe.m_benchmark = RecipeComponent{"Benchmark", "default"};
      def_recipe.m_stopping = RecipeComponent{"StoppingCriterion", "default"};
      def_recipe.m_sweep =
          SweepSpec{SweepStrategy::FullGrid,
                    {SweepAxis{"Case", "work_size", SweepHint{SweepPolicy::PowersOfTwo, "1", "1024", "1", {}}}}};
      def_recipe.m_description = "Default Recipe";
      protocol.m_recipes["default"] = def_recipe;
      Campaign default_campaign;
      default_campaign.m_name = "default";
      default_campaign.m_recipe = "default";
      for (const auto &backend : storage_manager->list_backends()) {
        default_campaign.m_backends.push_back({backend, "default"});
      }
      for (const auto &cases : storage_manager->list_components(ComponentType::CASE)) {
        default_campaign.m_cases.push_back({cases, "default"});
      }
      default_campaign.m_on_incompatible = OnIncompatible::Skip;
      protocol.m_campaigns.push_back(default_campaign);
      return protocol;
    }
  }; // namespace Orchestrator

} // namespace Baseliner
#endif // BASELINER_ORCHESTRATOR_HPP