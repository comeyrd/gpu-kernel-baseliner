#ifndef BASELINER_CORE_DURATIONS_HPP
#define BASELINER_CORE_DURATIONS_HPP
#include <baseliner/cli/Serializer.hpp>
#include <chrono>
#include <ostream>
#include <ratio>
#include <vector>

namespace Baseliner {
  using float_milliseconds = std::chrono::duration<float, std::milli>;

  inline auto sum(std::vector<float_milliseconds> &f_vector) -> float_milliseconds {
    float sum{0};
    for (auto &item : f_vector) {
      sum += item.count();
    }
    return float_milliseconds(sum);
  }
  template <typename T>
  struct ConfidenceInterval {
    T high;
    T low;
  };
  DESCRIBE_TEMPLATE(ConfidenceInterval, FIELD(high), FIELD(low))

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