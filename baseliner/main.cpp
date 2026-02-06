
#include <baseliner/Executable.hpp>
#include <baseliner/Serializer.hpp>
#include <iostream>
using namespace Baseliner;
__attribute__((weak)) int main(int argc, char **argv) {
  // std::cout << argc << argv[0] << std::endl;
  std::cout << "Baseliner" << std::endl;
  auto *manager = ExecutableManager::instance();
  auto &list = manager->getExecutables();

  std::cout << "[Baseliner] Total Registered Executables: " << list.size() << std::endl;

  if (list.empty()) {
    std::cout << "Warning: No kernels were registered. Check linker settings." << std::endl;
    return 0;
  }
  std::vector<Result> results_vector;
  for (auto &exe : list) {
    std::vector<Result> local_results = exe->run_all();
    // reserve space to avoid multiple reallocations
    results_vector.reserve(results_vector.size() + local_results.size());

    for (auto &res : local_results) {
      results_vector.emplace_back(std::move(res));
    }
    std::cout << "." << std::endl;
  }
  result_to_file(results_vector, "test.json");
  return 0;
};
