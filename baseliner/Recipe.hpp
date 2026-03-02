#ifndef BASELINER_RECIPE_HPP
#define BASELINER_RECIPE_HPP
#include <baseliner/managers/Components.hpp>
#include <iomanip>
#include <ios>
#include <optional>
#include <string>
#include <variant>
#include <vector>
namespace Baseliner {
  struct WithPreset {
    std::string m_name;
    std::string m_preset;
  };
  inline auto w_default_preset(const std::string &name) -> WithPreset {
    return WithPreset{name, std::string(DEFAULT_PRESET)};
  }
  struct Recipe {
    WithPreset m_backend;
    std::optional<WithPreset> m_suite;
    WithPreset m_benchmak;
    WithPreset m_case;
    WithPreset m_stats;
    WithPreset m_stopping;
  };

  inline auto create_recipe(const WithPreset &backend, const WithPreset &case_preset,
                            const std::optional<WithPreset> &suite = {},
                            const WithPreset &benchmark = w_default_preset(std::string(DEFAULT_BENCHMARK)),
                            const WithPreset &stats = w_default_preset(std::string(DEFAULT_STAT)),
                            const WithPreset &stopping = w_default_preset(std::string(DEFAULT_STOPPING))) -> Recipe {
    return Recipe{backend, suite, benchmark, case_preset, stats, stopping};
  }

  class RecipeManager {
  public:
    static void register_recipes(const std::vector<Recipe> &recipes) {
      for (const auto &recipe : recipes) {
        inner_get_recipes().push_back(recipe);
      }
    }
    static void register_recipe(const Recipe &recipe) {
      inner_get_recipes().push_back(recipe);
    }

    static auto get_recipes() -> std::vector<Recipe> {
      auto &recipes = inner_get_recipes();
      return recipes;
    }

  private:
    static auto inner_get_recipes() -> std::vector<Recipe> & {
      static std::vector<Recipe> recipes;
      return recipes;
    }
  };
  class RecipeRegistrar {
  public:
    explicit RecipeRegistrar(const Recipe &recipe) {
      RecipeManager::register_recipe(recipe);
    }
  };
  inline static void print_with_preset(std::ostream &oss, const WithPreset &wpreset, const std::string &name,
                                       int label_size) {
    oss << std::left << std::setw(label_size) << name << ": " << wpreset.m_name;
    if (wpreset.m_preset != DEFAULT_PRESET) {
      oss << " (" << wpreset.m_preset << ")";
    }
    oss << "\n";
  }
  inline static void print_recipe(std::ostream &oss, const Recipe &recipe) {
    int label_size = 25;
    oss << "\n_________________________\n";
    oss << "Executing Recipe :\n";
    print_with_preset(oss, recipe.m_backend, "Backend", label_size);
    print_with_preset(oss, recipe.m_case, "Case", label_size);
    if (recipe.m_benchmak.m_name != DEFAULT_BENCHMARK) {
      print_with_preset(oss, recipe.m_benchmak, "Benchmark", label_size);
    }
    if (recipe.m_stopping.m_name != DEFAULT_STOPPING) {
      print_with_preset(oss, recipe.m_stopping, "StoppingCriterion", label_size);
    }
    if (recipe.m_stats.m_name != DEFAULT_STAT) {
      print_with_preset(oss, recipe.m_stats, "Stats", label_size);
    }
    if (recipe.m_suite.has_value()) {
      auto val = recipe.m_suite.value();
      print_with_preset(oss, val, "Suite", label_size);
    }
    oss << "\n";
  }

#define BASELINER_REGISTER_RECIPE(Recipe) ATTRIBUTE_USED static Baseliner::RecipeRegistrar _registrar_##Recipe{Recipe};
} // namespace Baseliner
#endif //