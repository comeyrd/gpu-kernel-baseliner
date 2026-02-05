#ifndef BASELINER_SERIALIZER_HPP
#define BASELINER_SERIALIZER_HPP
#include <ostream>

namespace Baseliner {

  template <typename T>
  void serialize(std::ostream &os, const T &obj);

} // namespace Baseliner
#endif // BASELINER_SERIALIZER_HPP