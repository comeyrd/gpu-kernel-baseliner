#include <baseliner/Result.hpp>
#include <baseliner/Serializer.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Baseliner {
  void result_to_file(const std::vector<Result> &results, std::string filename) {
    std::ifstream infile(filename);
    const bool create_new_file = true;
    bool file_exists = false;
    if (infile.is_open()) {
      file_exists = true;
    }
    if (file_exists && create_new_file) {
      std::cout << filename << " already exists, but headers doesnt match. Writing in " << filename << ".new";
      filename = filename + ".new";
    }
    std::ofstream file;
    if (create_new_file) {
      file = std::ofstream(filename);
    } else {
      file = std::ofstream(filename, std::ios::app);
    }

    if (!file.is_open()) {
      throw std::runtime_error("Could not open file: " + filename);
    }
    serialize(file, results);
  }
} // namespace Baseliner