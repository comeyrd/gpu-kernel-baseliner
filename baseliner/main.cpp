
#include <baseliner/Executable.hpp>
#include <baseliner/Serializer.hpp>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

using namespace Baseliner;

std::string generate_uid() {
  using namespace std::chrono;
  auto now = duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);
  uint16_t rand_val = dis(gen);

  std::stringstream ss;
  ss << std::hex << std::setfill('0') << std::setw(8) << now << std::setw(2) << std::setw(4) << rand_val;
  return ss.str();
};

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
  }
  std::string filename = generate_uid() + ".json";
  result_to_file(results_vector, filename);
  return 0;
};
