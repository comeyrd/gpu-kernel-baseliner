#ifndef BASELINER_PROTOCOL_HPP
#define BASELINER_PROTOCOL_HPP
#include <baseliner/AxeSweeping.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/managers/Components.hpp>
#include <string>
#include <unordered_map>
#include <vector>
namespace Baseliner {
  enum class OnIncompatible : char {
    Skip,
    Error
  };

  struct RecipeComponent {
    std::string m_impl;
    std::optional<std::string> m_preset;
  };
  struct RecipeStat {
    std::string m_preset;
  };

  struct Recipe {
    std::string m_description;
    std::optional<RecipeComponent> m_benchmark;
    std::optional<RecipeComponent> m_stopping;
    std::optional<RecipeStat> m_stats;
    std::optional<SweepSpec> m_sweep;
  };

  struct CampaignOverrides {
    std::optional<RecipeComponent> m_benchmark;
    std::optional<RecipeComponent> m_stopping;
    std::optional<RecipeStat> m_stats;
  };

  struct Campaign {
    std::string m_name;
    std::string m_recipe;
    std::vector<RecipeComponent> m_cases;
    std::vector<RecipeComponent> m_backends;
    CampaignOverrides m_overrides;
    OnIncompatible m_on_incompatible;
  };

  struct Protocol {
    std::string m_baseliner_version;
    std::unordered_map<std::string, std::unordered_map<std::string, ComponentPreset>> m_presets;
    std::unordered_map<std::string, StatsPreset> m_stats_presets;
    std::unordered_map<std::string, Recipe> m_recipes;
    std::vector<Campaign> m_campaigns;
  };
} // namespace Baseliner
#endif // BASELINER_PROTOCOL_HPP