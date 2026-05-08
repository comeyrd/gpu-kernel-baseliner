#ifndef BASELINER_ORCHESTRATOR_ORCHESTRATOR_HPP
#define BASELINER_ORCHESTRATOR_ORCHESTRATOR_HPP
// #include <baseliner/RQ.hpp>
#include <baseliner/cli/CliHelper.hpp>
#include <baseliner/cli/Json.hpp>
#include <baseliner/core/GIT_VERSION.hpp>
#include <baseliner/core/Version.hpp>
#include <baseliner/orchestrator/Builder.hpp>
#include <baseliner/orchestrator/Plan.hpp>
#include <baseliner/orchestrator/Planner.hpp>
#include <baseliner/orchestrator/Protocol.hpp>
#include <baseliner/orchestrator/Report.hpp>
#include <baseliner/utils/Utils.hpp>
#include <vector>
namespace Baseliner {
  namespace Orchestrator {
    inline void load_presets(const Protocol &preset_protocol) {
      StorageManager::instance()->load_protocol_presets(preset_protocol);
    };
    [[nodiscard]] inline auto get_metadata_file() -> Metadata {
      return StorageManager::instance()->get_metadata();
    }
    inline auto run_benchmark_plan(const BenchmarkPlan &bench_plan, std::shared_ptr<Cli::CliPrinter> &printer,
                                   StorageManager *storage_manager = StorageManager::instance()) -> BenchmarkExecution {
      IBenchmarkFactory bench_factory = Builder::build(bench_plan, storage_manager);
      printer->print_benchmark_plan(bench_plan);
      std::shared_ptr<IBenchmark> bench = bench_factory();
      bench->set_printer(printer);
      BenchmarkReport bench_report = bench->run_benchmark();
      BenchmarkExecution benchmark_exec;
      benchmark_exec.benchmark_report = bench_report;
      benchmark_exec.plan = bench_plan;
      return benchmark_exec;
    };
    inline auto run_campaign_plan(const CampaignPlan &campaign_plan,
                                  StorageManager *storage_manager = StorageManager::instance()) -> CampaignReport {
      CampaignReport c_report;
      c_report.name = campaign_plan.name;
      c_report.recipe = campaign_plan.recipe;
      c_report.recipe_name = campaign_plan.recipe_name;
      std::shared_ptr<Cli::CliPrinter> printer = std::make_shared<Cli::CliPrinter>();
      printer->print_campaign_plan(campaign_plan);
      for (const auto &bench_plan : campaign_plan.benchmarks) {
        if (ExecutionController::exit_requested()) {
          break;
        }
        try {
          BenchmarkExecution bench_exec = run_benchmark_plan(bench_plan, printer, storage_manager);
          c_report.benchmark_runs[bench_plan.backend.impl][bench_plan.workload.impl] = bench_exec;
        } catch (const Error &e) {
          if (campaign_plan.on_incompatible != OnIncompatible::Skip) {
            throw;
          }
          std::cerr << "Warning : " << e.what() << "\n";
        }
      }
      return c_report;
    }

    inline auto run_protocol(const Protocol &protocol) -> Report {
      Report report;
      auto *storage_manager = StorageManager::instance();
      std::vector<CampaignPlan> plans = Planner::plan(protocol, storage_manager);
      for (const auto &plan : plans) {
        if (ExecutionController::exit_requested()) {
          break;
        }
        report.campaign_runs.push_back(run_campaign_plan(plan));
      }
      return report;
    };

    inline auto replay_runs(const Report &to_replay_report) -> Report {
      Report report;
      for (const CampaignReport &campaign_run : to_replay_report.campaign_runs) {
        CampaignPlan c_plan = campaign_plan_from_report(campaign_run);
        report.campaign_runs.push_back(run_campaign_plan(c_plan));
      }
      return report;
    };

    inline auto run_research_questions(const std::vector<std::string> &workload_names) -> Report {
      std::vector<std::string> backends = StorageManager::instance()->list_backends();
      std::vector<RecipeComponent> workload_components;
      workload_components.reserve(workload_names.size());
      std::vector<RecipeComponent> backends_components;
      backends_components.reserve(backends.size());

      for (const auto &workload_name : workload_names) {
        workload_components.push_back({workload_name, {}});
      }
      for (const auto &backend : backends) {
        backends_components.push_back({backend, {}});
      }
      // Protocol protocol = rq_protocol(RQSize::Medium, workload_components, backends_components);
      // return run_protocol(protocol);
      throw Errors::not_implemented("Research questions");
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
      protocol.baseliner_version = Version::string();
      protocol.presets = storage_manager->get_all_component_presets();
      protocol.stats_presets = storage_manager->get_all_stats_presets();
      Recipe def_recipe;
      def_recipe.stats = RecipeStat{"default"};
      def_recipe.benchmark = RecipeComponent{"Benchmark", "default"};
      def_recipe.stopping = RecipeComponent{"StoppingCriterion", "default"};
      def_recipe.sweep =
          SweepSpec{SweepStrategy::FullGrid,
                    {SweepAxis{"Workload", "work_size", SweepHint{SweepPolicy::PowersOfTwo, "1", "256", "1", {}}}}};
      def_recipe.description = "Default Recipe";
      protocol.recipes["default"] = def_recipe;
      Campaign default_campaign;
      default_campaign.name = "default";
      default_campaign.recipe = "default";
      for (const auto &backend : storage_manager->list_backends()) {
        default_campaign.backends.push_back({backend, "default"});
      }
      for (const auto &workloads : storage_manager->list_components(ComponentType::WORKLOAD)) {
        default_campaign.workloads.push_back({workloads, "default"});
      }
      default_campaign.on_incompatible = OnIncompatible::Skip;
      protocol.campaigns.push_back(default_campaign);
      return protocol;
    }
    inline auto run_default(std::string device) -> Report {
      Protocol protocol = get_default_protocol();
      auto *storage_manager = StorageManager::instance();
      auto backends = storage_manager->list_backends();
      for (const auto &backend : backends) {
        ComponentPreset preset = storage_manager->get_component_preset(backend, "default");
        preset.options["Backend"]["device"].value = device;
        protocol.presets[backend]["default"] = preset;
      }
      return run_protocol(protocol);
    }
    inline auto run_primbench_default(std::string device) -> Report {
      Protocol protocol = get_default_protocol();
      auto *storage_manager = StorageManager::instance();
      auto backends = storage_manager->list_backends();
      for (const auto &backend : backends) {
        ComponentPreset preset = storage_manager->get_component_preset(backend, "default");
        preset.options["Backend"]["device"].value = device;
        protocol.presets[backend]["default"] = preset;
      }
      auto set_opt = [](ComponentPreset &p, const std::string &key, const std::string &val) {
        p.options["Benchmark"][key] = {std::nullopt, val};
      };

      ComponentPreset primbench_preset;
      primbench_preset.description = "primbench-style: dynamic batch size, blocking kernel, warm/cool cycle, L2 flush";
      set_opt(primbench_preset, "validate_workload", "0");
      set_opt(primbench_preset, "min_gpu_temp", "50.000000");
      set_opt(primbench_preset, "max_gpu_temp", "60.000000");
      set_opt(primbench_preset, "warm_cool_timeout", "60");
      set_opt(primbench_preset, "warm_cool", "1");
      set_opt(primbench_preset, "warmup", "1");
      set_opt(primbench_preset, "flush", "1");
      set_opt(primbench_preset, "dynamic_batch", "1");
      set_opt(primbench_preset, "minimal_batch_duration", "10.000000");
      set_opt(primbench_preset, "batch_size", "1");
      set_opt(primbench_preset, "block", "1");
      set_opt(primbench_preset, "block_duration", "10000.000000");
      set_opt(primbench_preset, "block_queue_size", "64");
      protocol.presets["Benchmark"]["default"] = primbench_preset;
      return run_protocol(protocol);
    }
    inline auto run_nvbench_default(std::string device) -> Report {
      Protocol protocol = get_default_protocol();
      auto *storage_manager = StorageManager::instance();
      auto backends = storage_manager->list_backends();
      for (const auto &backend : backends) {
        ComponentPreset preset = storage_manager->get_component_preset(backend, "default");
        preset.options["Backend"]["device"].value = device;
        protocol.presets[backend]["default"] = preset;
      }
      auto set_opt = [](ComponentPreset &p, const std::string &key, const std::string &val) {
        p.options["Benchmark"][key] = {std::nullopt, val};
      };

      ComponentPreset nvbench_preset;
      nvbench_preset.description = "nvbench-style: fixed batch size, no blocking kernel, no active warm/cool, L2 flush";
      set_opt(nvbench_preset, "validate_workload", "0");
      set_opt(nvbench_preset, "min_gpu_temp", "60.000000");
      set_opt(nvbench_preset, "max_gpu_temp", "50.000000");
      set_opt(nvbench_preset, "warm_cool_timeout", "3");
      set_opt(nvbench_preset, "warm_cool", "0");
      set_opt(nvbench_preset, "warmup", "1");
      set_opt(nvbench_preset, "flush", "1");
      set_opt(nvbench_preset, "dynamic_batch", "0");
      set_opt(nvbench_preset, "minimal_batch_duration", "10.000000");
      set_opt(nvbench_preset, "batch_size", "1");
      set_opt(nvbench_preset, "block", "0");
      set_opt(nvbench_preset, "block_duration", "1000.000000");
      set_opt(nvbench_preset, "block_queue_size", "64");
      protocol.presets["Benchmark"]["default"] = nvbench_preset;
      return run_protocol(protocol);
    }
    inline auto get_minimal_protocol() -> Protocol {
      Protocol protocol;
      auto *storage_manager = StorageManager::instance();
      protocol.baseliner_version = Version::string();
      Recipe def_recipe;
      def_recipe.stats = {};
      def_recipe.benchmark = {};
      def_recipe.stopping = {};
      def_recipe.sweep =
          SweepSpec{SweepStrategy::FullGrid,
                    {SweepAxis{"Workload", "work_size", SweepHint{SweepPolicy::PowersOfTwo, "1", "256", "1", {}}}}};
      def_recipe.description = "Minimal recipe with everything kept to default";
      protocol.recipes["minimal"] = def_recipe;
      Campaign default_campaign;
      default_campaign.name = "minimal";
      default_campaign.recipe = "minimal";
      for (const auto &backend : storage_manager->list_backends()) {
        default_campaign.backends.push_back({backend, {}});
      }
      for (const auto &workloads : storage_manager->list_components(ComponentType::WORKLOAD)) {
        default_campaign.workloads.push_back({workloads, {}});
      }
      default_campaign.on_incompatible = OnIncompatible::Skip;
      protocol.campaigns.push_back(default_campaign);
      return protocol;
    }
  }; // namespace Orchestrator

} // namespace Baseliner
#endif // BASELINER_ORCHESTRATOR_HPP