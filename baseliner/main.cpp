
#include <baseliner/Manager.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/Suite.hpp>
#include <baseliner/Task.hpp>
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

__attribute__((weak)) auto main(int argc, char **argv) -> int { // NOLINT
  // std::cout << argc << argv[0] << std::endl;
  std::cout << "Baseliner" << "\n";
  auto *manager = TaskManager::instance();
  std::string backend;
  std::string selected_benchmark;
  std::string selected_case;
  std::string added_stat;
  std::cout << "Backends \n";
  for (auto name : BackendRegistry::list_backends()) {
    std::cout << name << "\n";
    backend = name;
  }
  auto device_manager = BackendRegistry::get(backend);
  std::cout << "Cases : \n";
  for (auto name : device_manager->list_cases()) {
    std::cout << name << "\n";
    selected_case = name;
  }
  std::cout << "Benchmarks \n";
  for (auto name : device_manager->list_benchmarks()) {
    std::cout << name << "\n";
    selected_benchmark = name;
  }
  auto *stat_dict = Stats::StatsDictionnary::instance();
  std::cout << "Stats \n";
  for (auto name : stat_dict->list_stats()) {
    std::cout << name << "\n";
    added_stat = name;
  }

  auto bench = device_manager->get_benchmark_with_case(selected_benchmark, selected_case);
  // bench->add_stats({"Median", "SortedExecutionTimeVector", "Q1"});
  BASELINER_REGISTER_TASK(bench);
  const auto &list = manager->get_tasks();

  std::cout << "[Baseliner] Total Registered Executables: " << list.size() << "\n";

  if (list.empty()) {
    std::cout << "Warning: No kernels were registered. Check linker settings." << "\n";
    return 0;
  }
  std::vector<Result> results_vector;
  for (const auto &exe : list) {
    std::vector<Result> local_results = exe->run_all();
    std::cout << exe->print_console(local_results);
    // reserve space to avoid multiple reallocations
    results_vector.reserve(results_vector.size() + local_results.size());

    for (auto &res : local_results) {
      results_vector.emplace_back(std::move(res));
    }
  }
  const std::string filename = generate_uid() + ".json";
  result_to_file(results_vector, filename);
  return 0;
};
