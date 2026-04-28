#include <baseliner/core/Error.hpp>

#include <baseliner/core/Options.hpp>

#include <baseliner/orchestrator/Plan.hpp>
#include <baseliner/orchestrator/Planner.hpp>
#include <baseliner/orchestrator/Protocol.hpp>
#include <baseliner/registry/Components.hpp>

namespace Baseliner::Planner {

  class PresetCascader {
  public:
    explicit PresetCascader(
        const std::unordered_map<std::string, std::unordered_map<std::string, ComponentPreset>> &component_presets,
        const std::unordered_map<std::string, StatsPreset> &stats_presets, const StorageManager *storage_manager)
        : component_presets(component_presets),
          stats_presets(stats_presets),
          storage_manager(storage_manager) {};

    [[nodiscard]] auto cascade(const std::optional<RecipeComponent> &override_comp,
                               const std::optional<RecipeComponent> &base_comp, const ComponentType &type) const
        -> PlannedComponent {
      if (override_comp.has_value())
        return cascade(override_comp.value());
      if (base_comp.has_value())
        return cascade(base_comp.value());
      return cascade(RecipeComponent{component_to_string(type), {}});
    }

    [[nodiscard]] auto cascade(const std::optional<RecipeComponent> &wanted_component, const ComponentType &type) const
        -> PlannedComponent {
      RecipeComponent current_component;
      if (wanted_component.has_value()) {
        current_component = wanted_component.value();
      } else {
        current_component = {component_to_string(type), {}};
      }
      return cascade(current_component);
    }

    [[nodiscard]] auto cascade(const RecipeComponent &wanted_component) const -> PlannedComponent {
      OptionsMap found_omap{};
      std::string current_preset = std::string(DEFAULT_PRESET);
      if (wanted_component.preset.has_value()) {
        current_preset = wanted_component.preset.value();
      }
      if (component_presets.find(wanted_component.impl) != component_presets.end()) {
        const auto &middle_map = component_presets.at(wanted_component.impl);
        if (middle_map.find(current_preset) != middle_map.end()) {
          found_omap = middle_map.at(current_preset).options;
        }
      }
      try {
        const OptionsMap temp_omap =
            storage_manager->get_component_preset(wanted_component.impl, current_preset).options;
        found_omap = Options::merge(temp_omap, found_omap);
      } catch (const Error &e) {
        if (found_omap.empty()) {
          throw e;
        } else {
          std::cerr << "Encountered error : " << e.what() << "\n";
        }
      }
      return {wanted_component.impl, current_preset, found_omap};
    }

    [[nodiscard]] auto cascade(const std::optional<RecipeStat> &override_stat,
                               const std::optional<RecipeStat> &base_stat) const -> PlannedStat {
      if (override_stat.has_value()) {
        return cascade(override_stat.value());
      }
      if (base_stat.has_value()) {
        return cascade(base_stat.value());
      }
      return cascade(RecipeStat{std::string(DEFAULT_STAT)});
    }

    [[nodiscard]] auto cascade(const RecipeStat &wanted_stat) const -> PlannedStat {
      StatsPreset found_preset{};
      if (stats_presets.find(wanted_stat.preset) != stats_presets.end()) {
        found_preset = stats_presets.at(wanted_stat.preset);
      }
      try {
        const OptionsMap temp_omap = storage_manager->get_stats_preset(wanted_stat.preset).stat_options;
        found_preset.stat_options = Options::merge(temp_omap, found_preset.stat_options);
      } catch (const Error &e) {
        if (found_preset.stat_names.empty()) {
          throw e;
        } else {
          std::cerr << "Encountered error : " << e.what() << "\n";
        }
      }
      return PlannedStat{wanted_stat.preset, found_preset.stat_names, found_preset.stat_options};
    }

    [[nodiscard]] auto cascade(const std::optional<RecipeStat> &wanted_stat) const -> PlannedStat {
      if (wanted_stat.has_value()) {
        return cascade(wanted_stat.value());
      }
      return cascade(RecipeStat{std::string(DEFAULT_STAT)});
    }

  private:
    const std::unordered_map<std::string, std::unordered_map<std::string, ComponentPreset>> &component_presets;
    const std::unordered_map<std::string, StatsPreset> &stats_presets;
    const StorageManager *storage_manager;
  };

  auto plan(const Protocol &protocol, const StorageManager *storage_manager) -> std::vector<CampaignPlan> {
    std::vector<CampaignPlan> campaigns{};
    PresetCascader cascader(protocol.presets, protocol.stats_presets, storage_manager);
    for (const Campaign &current_campaign : protocol.campaigns) {
      CampaignPlan campaign_plan;
      campaign_plan.name = current_campaign.name;
      campaign_plan.recipe_name = current_campaign.recipe;
      if (protocol.recipes.find(current_campaign.recipe) == protocol.recipes.end()) {
        throw Errors::not_found("Recipe", current_campaign.recipe);
      }
      const Recipe &wanted_recipe = protocol.recipes.at(current_campaign.recipe);
      campaign_plan.recipe = wanted_recipe;

      const auto &ovr = current_campaign.overrides.value_or(CampaignOverrides{});

      BenchmarkPlan bench_plan;
      bench_plan.benchmark = cascader.cascade(ovr.benchmark, wanted_recipe.benchmark, ComponentType::BENCHMARK);
      bench_plan.stopping = cascader.cascade(ovr.stopping, wanted_recipe.stopping, ComponentType::STOPPING);
      bench_plan.stats = cascader.cascade(ovr.stats, wanted_recipe.stats);
      bench_plan.sweep = wanted_recipe.sweep;
      campaign_plan.on_incompatible = current_campaign.on_incompatible;

      for (const RecipeComponent &current_workload : current_campaign.workloads) {
        for (const RecipeComponent &current_backend : current_campaign.backends) {
          try {
            bench_plan.backend = cascader.cascade(current_backend);
            bench_plan.workload = cascader.cascade(current_workload);
            campaign_plan.benchmarks.push_back(bench_plan);
          } catch (const Error &e) {
            if (e.code() == ErrorCode::BackendWorkloadBenchmarkNotFound || e.code() == ErrorCode::NotFound) {
              if (current_campaign.on_incompatible == OnIncompatible::Skip) {
                std::cout << "Warning" << e.what() << "\n";
                continue;
              }
              if (current_campaign.on_incompatible == OnIncompatible::Error) {
                throw e;
              }
            } else {
              throw e;
            }
          }
        }
      }
      campaigns.push_back(campaign_plan);
    }
    return campaigns;
  };

} // namespace Baseliner::Planner