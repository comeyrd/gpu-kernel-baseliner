#ifndef BASELINER_SERIALIZER_HPP
#define BASELINER_SERIALIZER_HPP
#include <baseliner/Error.hpp>
#include <fstream>
#include <iostream>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Baseliner {

  template <typename T>
  void serialize(std::ostream &oss, const T &obj);
  template <typename T>
  void de_serialize(std::istream &iss, T &obj);

  template <typename T>
  auto from_file(const std::string &filename) -> T {
    T object;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
      throw Errors::file_read_error(filename);
    }
    de_serialize(infile, object);
    return object;
  };

  template <typename T>
  void to_file(const T &object, const std::string &filename) {
    std::ifstream infile(filename);
    auto file = std::ofstream(filename, std::ios::trunc);
    if (!file.is_open()) {
      throw Errors::file_write_error(filename);
    }
    serialize(file, object);
  }

} // namespace Baseliner
#endif // BASELINER_SERIALIZER_HPP