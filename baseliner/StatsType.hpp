#ifndef BASELINER_STATS_TYPE_HPP
#define BASELINER_STATS_TYPE_HPP
namespace Baseliner {
  template <typename T>

  struct ConfidenceInterval {
    T high;
    T low;
  };

} // namespace Baseliner
#endif // BASELINER_STATS_TYPE_HPP