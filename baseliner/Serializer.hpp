#ifndef BASELINER_SERIALIZER_HPP
#define BASELINER_SERIALIZER_HPP

#include <istream>
#include <ostream>

namespace Baseliner {

  template <typename T>
  void serialize(std::ostream &oss, const T &obj);
  template <typename T>
  void de_serialize(std::istream &iss, T &obj);

} // namespace Baseliner
#endif // BASELINER_SERIALIZER_HPP