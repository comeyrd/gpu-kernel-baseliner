
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

static auto generate_uid() -> std::string { // NOLINT
  using namespace std::chrono;
  auto now = duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
  static std::random_device rd; // NOLINT
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF); // NOLINT
  const uint16_t rand_val = dis(gen);

  std::stringstream stringstream;
  stringstream << std::hex << std::setfill('0') << std::setw(8) << now // NOLINT
               << std::setw(2) << std::setw(4) << rand_val;
  return stringstream.str();
};

__attribute__((weak)) auto main(int argc, char **argv) -> int { // NOLINT
  std::cout << "Baseliner" << "\n";
  auto recipes = RecipeManager::get_recipes();
  std::cout << "[Baseliner] Total Registered Recipes: " << recipes.size() << "\n";

  std::vector<Result> results;

  Manager *manager = Manager::instance();
  metadata_to_file(manager->generate_metadata(), "metadata.json");
  config_to_file(manager->generate_example_config(), "default-config.json");
  Config config;
  file_to_config(config, "config.json");
  manager->add_presets(config.m_presets);
  for (const auto &recipe : config.m_recipes) {
    RecipeManager::register_recipe(recipe);
  }
  for (auto &recipe : RecipeManager::get_recipes()) {
    auto bench_or_suite = manager->build_recipe(recipe);
    if (std::holds_alternative<std::function<std::shared_ptr<IBenchmark>()>>(bench_or_suite)) {
      auto bench = std::get<std::function<std::shared_ptr<IBenchmark>()>>(bench_or_suite);
      results.push_back(bench()->run());
    } else if (std::holds_alternative<std::function<std::shared_ptr<ISuite>()>>(bench_or_suite)) {
      auto suite = std::get<std::function<std::shared_ptr<ISuite>()>>(bench_or_suite);
      std::vector<Result> temp_result = suite()->run_all();
      results.insert(results.end(), temp_result.begin(), temp_result.end());
    }
  }
  const std::string filename = generate_uid() + ".json";

  result_to_file(results, filename);
  return 0;
};
