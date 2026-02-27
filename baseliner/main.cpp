
#include "baseliner/Handler.hpp"
#include <baseliner/Recipe.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/Suite.hpp>
#include <baseliner/managers/Manager.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <utility>
#include <variant>
#include <vector>
using namespace Baseliner;

__attribute__((weak)) auto main(int argc, char **argv) -> int { // NOLINT
  std::cout << "Baseliner" << "\n";

  Manager *manager = Manager::instance();
  metadata_to_file(manager->generate_metadata(), "metadata.json");
  config_to_file(manager->generate_example_config(), "default-config.json");
  Config config;
  file_to_config(config, "config.json");
  manager->add_presets(config.m_presets);
  RecipeManager::register_recipes(config.m_recipes);
  Handler handler;
  std::vector<Recipe> recipes = RecipeManager::get_recipes();
  std::cout << "[Baseliner] Total Registered Recipes: " << recipes.size() << "\n";

  Result result = handler.run_recipes(recipes);
  const std::string filename = generate_uid() + ".json";
  result_to_file(result, filename);
  return 0;
};
