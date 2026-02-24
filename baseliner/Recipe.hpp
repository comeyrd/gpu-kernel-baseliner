#ifndef BASELINER_RECIPE_HPP
#define BASELINER_RECIPE_HPP
#include <optional>
#include <string>
#include <vector>
namespace Baseliner {
  struct Recipe {
    std::string m_backend;
    std::optional<std::string> m_suite;
    std::string m_benchmak;
    std::string m_case;
    std::vector<std::string> m_stats;
    std::string m_stopping;
  };

  class RecipeManager {
  public:
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

#define BASELINER_REGISTER_RECIPE(Recipe) ATTRIBUTE_USED static Baseliner::RecipeRegistrar _registrar_##Recipe{Recipe};
} // namespace Baseliner
#endif //