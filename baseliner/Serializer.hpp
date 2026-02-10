#ifndef BASELINER_SERIALIZER_HPP
#define BASELINER_SERIALIZER_HPP
#include <baseliner/Result.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace Baseliner {

  template <typename T>
  void serialize(std::ostream &oss, const T &obj);

  void result_to_file(const std::vector<Result> &results, std::string filename);
} // namespace Baseliner
#endif // BASELINER_SERIALIZER_HPP