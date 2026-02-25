
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

static void list_things() {};

__attribute__((weak)) auto main(int argc, char **argv) -> int { // NOLINT
  std::cout << "Baseliner" << "\n";
  auto recipes = RecipeManager::get_recipes();
  std::cout << "[Baseliner] Total Registered Recipes: " << recipes.size() << "\n";
  std::vector<Result> results;
  if (recipes.empty()) {
    list_things();
  }
  for (auto &recipe : RecipeManager::get_recipes()) {
  }
  const std::string filename = generate_uid() + ".json";

  result_to_file(results, filename);
  return 0;
};
