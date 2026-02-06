#ifndef BASELINER_SERIALIZER_HPP
#define BASELINER_SERIALIZER_HPP
#include <baseliner/Result.hpp>
#include <ostream>

namespace Baseliner {

  template <typename T>
  void serialize(std::ostream &os, const T &obj);

  void result_to_file(std::vector<Result>, std::string filename);
} // namespace Baseliner
#endif // BASELINER_SERIALIZER_HPP