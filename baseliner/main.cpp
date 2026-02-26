
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

static void list_things(std::ostream &oss) {
  Manager *manager = Manager::instance();
  oss << "Available Benchmarks :\n";
  for (auto [bench, presets] : manager->list_benchmark_presets()) {
    oss << bench << "\n";
  }
  oss << "Available Stopping Criterion :\n";
  for (auto [stopping, presets] : manager->list_stopping_presets()) {
    oss << stopping << "\n";
  }
  oss << "Available Suite :\n";
  for (auto [suite, presets] : manager->list_suite_presets()) {
    oss << suite << "\n";
  }
  oss << "Available Cases :\n";
  for (auto [cases, presets] : manager->list_cases_presets()) {
    oss << cases << "\n";
  }
  oss << "Available Stats Presets :\n";
  for (auto [stats, presets] : manager->list_general_stats_presets()) {
    oss << stats << "\n";
  }
  oss << "Available General Stats:\n";
  for (auto stat : manager->list_stats()) {
    oss << stat << "\n";
  }
  oss << "\n\nBackends : \n";
  for (const auto &[name, backend] : manager->list_backends()) {
    oss << name << "\n";
    oss << "Cases :\n";
    for (const auto &single_case : backend->list_device_cases()) {
      oss << single_case << "\n";
    }
    oss << "Benchmark :\n";
    for (const auto &bench : backend->list_device_benchmarks()) {
      oss << bench << "\n";
    }
    oss << "Stat :\n";
    for (const auto &stat : backend->list_device_stats()) {
      oss << stat << "\n";
    }
    oss << "---------------\n";
  }
};

__attribute__((weak)) auto main(int argc, char **argv) -> int { // NOLINT
  std::cout << "Baseliner" << "\n";
  auto recipes = RecipeManager::get_recipes();
  std::cout << "[Baseliner] Total Registered Recipes: " << recipes.size() << "\n";

  std::vector<Result> results;
  if (recipes.empty()) {
    list_things(std::cout);
  }
  Manager *manager = Manager::instance();
  metadata_to_file(manager->generate_metadata(), "metadata.json");

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
