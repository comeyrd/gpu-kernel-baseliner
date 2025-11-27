#ifndef DURATIONS_HPP
#define DURATIONS_HPP
#include <chrono>
#include <ostream>
#include <vector>
using float_milliseconds = std::chrono::duration<float, std::milli>;
inline std::ostream &operator<<(std::ostream &os, const float_milliseconds &duration) {
  float count = duration.count();
  os << count;
  return os;
}
inline std::ostream &operator<<(std::ostream &os, const std::vector<float_milliseconds> &duration_vector) {
  os << "[ ";
  for (const float_milliseconds &item : duration_vector) {
    os << item << " ,";
  }
  os << " ]";
  return os;
}
#endif // DURATIONS_HPP