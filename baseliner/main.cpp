
#include <baseliner/Recipe.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/Suite.hpp>
#include <baseliner/managers/BackendManager.hpp>
#include <baseliner/managers/BenchmarkCaseManager.hpp>
#include <baseliner/managers/StoppingManager.hpp>
#include <baseliner/managers/SuiteManager.hpp>

#include <baseliner/backend/cuda/CudaBackend.hpp>
#include <baseliner/stats/StatsDictionnary.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <utility>
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

static void list_things() {
  auto backends = BackendManager::list_backends();
  std::cout << "Number of available Backends : " << backends.size() << "\n ";
  for (auto backend_name : backends) {
    std::cout << backend_name << "\n";
    auto *device_manager = BackendManager::get(backend_name);
    std::cout << "Cases : \n";
    for (auto name : device_manager->list_cases()) {
      std::cout << name << "\n";
    }
    std::cout << "Benchmarks \n";
    for (auto name : device_manager->list_benchmarks()) {
      std::cout << name << "\n";
    }
  }
  std::cout << "------------------\n";
  std::cout << "Suites \n";
  for (auto name : SuiteManager::list_suites()) {
    std::cout << name << "\n";
  }
  auto *stat_dict = Stats::StatsDictionnary::instance();
  std::cout << "Available Stats :\n";
  for (auto name : stat_dict->list_stats()) {
    std::cout << name << "\n";
  }
  auto *s_manager = StoppingManager::instance();
  std::cout << "Available Stopping Criterions :\n";
  for (auto name : s_manager->list_stopping()) {
    std::cout << name << "\n";
  }

  std::cout << "Recipes : \n";
  for (auto &recipe : RecipeManager::get_recipes()) {
    std::cout << "backend : " << recipe.m_backend;
    std::cout << "case : " << recipe.m_case;
    std::cout << "benchmark : " << recipe.m_benchmak;
    if (recipe.m_suite.has_value()) {
      std::cout << "suite : " << recipe.m_suite.value();
    }
  }
};

__attribute__((weak)) auto main(int argc, char **argv) -> int { // NOLINT
  std::cout << "Baseliner" << "\n";
  auto recipes = RecipeManager::get_recipes();
  std::cout << "[Baseliner] Total Registered Recipes: " << recipes.size() << "\n";
  std::vector<Result> results;
  if (recipes.empty()) {
    list_things();
  }
  for (auto &recipe : RecipeManager::get_recipes()) {
    auto *backend = BackendManager::get(recipe.m_backend);
    auto benchmark = backend->get_benchmark_with_case(recipe.m_benchmak, recipe.m_case);
    auto bench_w_stopping = [benchmark, recipe]() -> std::shared_ptr<IBenchmark> {
      auto bench = benchmark();
      auto func = StoppingManager::instance()->get_stopping_recipe(recipe.m_stopping);
      bench->set_stopping_criterion(func);
      return bench;
    };
    if (recipe.m_suite.has_value()) {
      auto suite = std::make_shared<SingleAxeSuite>();
      suite->set_benchmark(bench_w_stopping);
      suite->set_axe(SuiteManager::get_suite(recipe.m_suite.value()));
      auto temp_res = suite->run_all();
      results.insert(results.end(), temp_res.begin(), temp_res.end());
    } else {
      auto bench = bench_w_stopping();
      results.push_back(bench->run());
    }
  }
  const std::string filename = generate_uid() + ".json";

  result_to_file(results, filename);
  return 0;
};
