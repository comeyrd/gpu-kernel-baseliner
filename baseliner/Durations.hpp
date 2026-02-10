#ifndef DURATIONS_HPP
#define DURATIONS_HPP
#include <chrono>
#include <ostream>
#include <ratio>
#include <vector>
namespace Baseliner {
  using float_milliseconds = std::chrono::duration<float, std::milli>;

} // namespace Baseliner
inline auto operator<<(std::ostream &outputStream, const Baseliner::float_milliseconds &duration) -> std::ostream & {
  const float count = duration.count();
  outputStream << count;
  return outputStream;
}

inline auto operator<<(std::ostream &outputStream, const std::vector<Baseliner::float_milliseconds> &duration_vector)
    -> std::ostream & {
  outputStream << "[ ";
  for (const Baseliner::float_milliseconds &item : duration_vector) {
    outputStream << item << " ,";
  }
  outputStream << " ]";
  return outputStream;
}
#endif // DURATIONS_HPP