
#include <baseliner/Manager.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/Suite.hpp>
#include <baseliner/Task.hpp>
#include <baseliner/backend/cuda/CudaBackend.hpp>
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
  auto *cuda_manager = Manager<Backend::CudaBackend>::instance();
  std::cout << "Cases : \n";
  for (auto name : cuda_manager->get_cases_names()) {
    std::cout << name << "\n";
  }
  for (auto name : cuda_manager->get_benchmarks_names()) {
    std::cout << name << "\n";
  }
  auto case_ = cuda_manager->get_case("ComputationKernel");
  auto bench = cuda_manager->get_benchmark("CudaBenchmark");
  bench->set_case(case_);
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
