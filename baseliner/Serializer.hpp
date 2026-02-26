#ifndef BASELINER_SERIALIZER_HPP
#define BASELINER_SERIALIZER_HPP
#include "baseliner/ConfigFile.hpp"
#include <baseliner/Metadata.hpp>
#include <baseliner/Result.hpp>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace Baseliner {

  template <typename T>
  void serialize(std::ostream &oss, const T &obj);
  template <typename T>
  void de_serialize(std::istream &iss, T &obj);
  void result_to_file(const std::vector<Result> &results, std::string filename);
  void metadata_to_file(const Metadata &metadata, std::string filename);
  void config_to_file(const Config &config, const std::string &filename);
  void file_to_config(Config &config, const std::string &filename);
} // namespace Baseliner
#endif // BASELINER_SERIALIZER_HPP