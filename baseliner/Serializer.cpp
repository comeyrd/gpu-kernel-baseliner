#include <baseliner/ConfigFile.hpp>
#include <baseliner/Metadata.hpp>

#include <baseliner/Serializer.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Baseliner {
  void result_to_file(const Result &results, std::string filename) {
    std::ifstream infile(filename);
    const bool create_new_file = true;
    bool file_exists = false;
    if (infile.is_open()) {
      file_exists = true;
    }
    if (file_exists && create_new_file) {
      std::cout << filename << " already exists, Writing in " << filename << ".new";
      filename = filename + ".new";
    }
    std::ofstream file;
    if (create_new_file) {
      file = std::ofstream(filename);
    } else {
      file = std::ofstream(filename, std::ios::app);
    }

    if (!file.is_open()) {
      throw Errors::file_write_error(filename);
    }
    serialize(file, results);
  }
  void file_to_result(Result &result, const std::string filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
      throw Errors::file_read_error(filename);
    }
    de_serialize(infile, result);
  }
  void metadata_to_file(const Metadata &metadata, std::string filename) {
    std::ifstream infile(filename);
    const bool create_new_file = true;
    auto file = std::ofstream(filename, std::ios::trunc);

    if (!file.is_open()) {
      throw Errors::file_write_error(filename);
    }
    serialize(file, metadata);
  }
  void config_to_file(const Config &config, const std::string &filename) {
    std::ifstream infile(filename);
    const bool create_new_file = true;
    auto file = std::ofstream(filename, std::ios::trunc);

    if (!file.is_open()) {
      throw Errors::file_write_error(filename);
    }
    serialize(file, config);
  }
  void file_to_config(Config &config, const std::string &filename) {
    std::ifstream infile(filename);

    if (!infile.is_open()) {
      throw Errors::file_read_error(filename);
    }
    de_serialize(infile, config);
  }
} // namespace Baseliner