#ifndef BASELINER_ORCHESTRATOR_PROTOCOL_HPP
#define BASELINER_ORCHESTRATOR_PROTOCOL_HPP
#include <baseliner/cli/Serializer.hpp>
#include <baseliner/registry/Components.hpp>
#include <baseliner/specs/AxeSweeping.hpp>
#include <string>
#include <unordered_map>
#include <vector>
namespace Baseliner {
  enum class OnIncompatible : char {
    Skip,
    Error
  };
  DESCRIBE_ENUM(OnIncompatible, ENUM_VALUE(Skip), ENUM_VALUE(Error))
  struct RecipeComponent {
    std::string impl;
    std::optional<std::string> preset;
  };
  DESCRIBE(RecipeComponent, FIELD(impl), FIELD(preset))
  struct RecipeStat {
    std::string preset;
  };
  DESCRIBE(RecipeStat, FIELD(preset))

  struct Recipe {
    std::string description;
    std::optional<RecipeComponent> benchmark;
    std::optional<RecipeComponent> stopping;
    std::optional<RecipeStat> stats;
    std::optional<SweepSpec> sweep;
  };
  DESCRIBE(Recipe, FIELD(description), FIELD(benchmark), FIELD(stopping), FIELD(stats), FIELD(sweep))

  struct CampaignOverrides {
    std::optional<RecipeComponent> benchmark;
    std::optional<RecipeComponent> stopping;
    std::optional<RecipeStat> stats;
  };
  DESCRIBE(CampaignOverrides, FIELD(benchmark), FIELD(stopping), FIELD(stats))
  struct Campaign {
    std::string name;
    std::string recipe;
    std::vector<RecipeComponent> workloads;
    std::vector<RecipeComponent> backends;
    std::optional<CampaignOverrides> overrides;
    OnIncompatible on_incompatible;
  };
  DESCRIBE(Campaign, FIELD(name), FIELD(recipe), FIELD(workloads), FIELD(backends), FIELD(overrides),
           FIELD(on_incompatible))

  struct Protocol {
    std::string baseliner_version;
    std::unordered_map<std::string, std::unordered_map<std::string, ComponentPreset>> presets;
    std::unordered_map<std::string, StatsPreset> stats_presets;
    std::unordered_map<std::string, Recipe> recipes;
    std::vector<Campaign> campaigns;
  };
  DESCRIBE(Protocol, FIELD(baseliner_version), FIELD(presets), FIELD(stats_presets), FIELD(recipes), FIELD(campaigns))

} // namespace Baseliner
#endif // BASELINER_PROTOCOL_HPP