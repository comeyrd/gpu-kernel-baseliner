#include <baseliner/Error.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Output.hpp>
#include <baseliner/Planner.hpp>
#include <baseliner/Protocol.hpp>
#include <baseliner/managers/Components.hpp>
namespace Baseliner::Planner {

  class PresetCascader {
  public:
    explicit PresetCascader(
        const std::unordered_map<std::string, std::unordered_map<std::string, ComponentPreset>> &component_presets,
        const std::unordered_map<std::string, StatsPreset> &stats_presets, const StorageManager *storage_manager)
        : m_component_presets(component_presets),
          m_stats_presets(stats_presets),
          m_storage_manager(storage_manager) {};

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
      if (wanted_component.m_preset.has_value()) {
        current_preset = wanted_component.m_preset.value();
      }
      if (m_component_presets.find(wanted_component.m_impl) != m_component_presets.end()) {
        const auto &middle_map = m_component_presets.at(wanted_component.m_impl);
        if (middle_map.find(current_preset) != middle_map.end()) {
          found_omap = middle_map.at(current_preset).m_options;
        }
      }
      try {
        const OptionsMap temp_omap =
            m_storage_manager->get_component_preset(wanted_component.m_impl, current_preset).m_options;
        found_omap = Options::merge(temp_omap, found_omap);
      } catch (const Error &e) {
        if (found_omap.empty()) {
          throw e;
        }
      }
      return {wanted_component.m_impl, current_preset, found_omap};
    }
    [[nodiscard]] auto cascade(const RecipeStat &wanted_stat) const -> PlannedStat {
      StatsPreset found_preset{};
      if (m_stats_presets.find(wanted_stat.m_preset) != m_stats_presets.end()) {
        found_preset = m_stats_presets.at(wanted_stat.m_preset);
      }
      try {
        const OptionsMap temp_omap = m_storage_manager->get_stats_preset(wanted_stat.m_preset).m_stat_options;
        found_preset.m_stat_options = Options::merge(temp_omap, found_preset.m_stat_options);
      } catch (const Error &e) {
        if (found_preset.m_stat_names.empty()) {
          throw e;
        }
      }
      return PlannedStat{wanted_stat.m_preset, found_preset.m_stat_names, found_preset.m_stat_options};
    }
    [[nodiscard]] auto cascade(const std::optional<RecipeStat> &wanted_stat) const -> PlannedStat {
      if (wanted_stat.has_value()) {
        return cascade(wanted_stat.value());
      }
      return cascade(RecipeStat{std::string(DEFAULT_STAT)});
    }

  private:
    const std::unordered_map<std::string, std::unordered_map<std::string, ComponentPreset>> &m_component_presets;
    const std::unordered_map<std::string, StatsPreset> &m_stats_presets;
    const StorageManager *m_storage_manager;
  };

  auto plan(const Protocol &protocol, const StorageManager *storage_manager) -> std::vector<Plan> {
    std::vector<Plan> plan_vector{};
    PresetCascader cascader(protocol.m_presets, protocol.m_stats_presets, storage_manager);
    for (const Campaign &current_campaign : protocol.m_campaigns) {
      Plan current_plan;
      current_plan.m_campaign_name = current_campaign.m_name;
      current_plan.m_recipe_name = current_campaign.m_recipe;
      if (protocol.m_recipes.find(current_campaign.m_recipe) == protocol.m_recipes.end()) {
        throw Errors::not_found("Recipe", current_campaign.m_recipe);
      }
      const Recipe &wanted_recipe = protocol.m_recipes.at(current_campaign.m_recipe);
      current_plan.m_benchmark = cascader.cascade(wanted_recipe.m_benchmark, ComponentType::BENCHMARK);
      current_plan.m_stopping = cascader.cascade(wanted_recipe.m_stopping, ComponentType::STOPPING);
      current_plan.m_stats = cascader.cascade(wanted_recipe.m_stats);
      current_plan.m_sweep = wanted_recipe.m_sweep;
      current_plan.m_on_incompatible = current_campaign.m_on_incompatible;
      for (const RecipeComponent &current_case : current_campaign.m_cases) {
        for (const RecipeComponent &current_backend : current_campaign.m_backends) {
          try {
            current_plan.m_backend = cascader.cascade(current_backend);
            current_plan.m_case = cascader.cascade(current_case);
            plan_vector.push_back(current_plan);
          } catch (const Error &e) {
            if (e.code() == ErrorCode::BackendCaseBenchmarkNotFound) {
              if (current_campaign.m_on_incompatible == OnIncompatible::Skip) {
                std::cout << "Warning" << e.what() << "\n";
                continue;
              }
              if (current_campaign.m_on_incompatible == OnIncompatible::Error) {
                throw e;
              }
            } else {
              throw e;
            }
          }
        }
      }
    }
    return plan_vector;
  };

} // namespace Baseliner::Planner