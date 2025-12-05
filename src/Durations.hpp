#ifndef DURATIONS_HPP
#define DURATIONS_HPP
#include <chrono>
#include <ostream>
#include <vector>
namespace Baseliner {
  using float_milliseconds = std::chrono::duration<float, std::milli>;

} // namespace Baseliner
inline std::ostream &operator<<(std::ostream &os, const Baseliner::float_milliseconds &duration) {
  float count = duration.count();
  os << count;
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const std::vector<Baseliner::float_milliseconds> &duration_vector) {
  os << "[ ";
  for (const Baseliner::float_milliseconds &item : duration_vector) {
    os << item << " ,";
  }
  os << " ]";
  return os;
}
#endif // DURATIONS_HPP