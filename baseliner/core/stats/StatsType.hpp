#ifndef BASELINER_CORE_STATS_STATSTYPE_HPP
#define BASELINER_CORE_STATS_STATSTYPE_HPP
#include <baseliner/cli/Serializer.hpp>
namespace Baseliner {
  template <typename T>
  struct ConfidenceInterval {
    T high;
    T low;
  };
  DESCRIBE_TEMPLATE(ConfidenceInterval, FIELD(high), FIELD(low))
} // namespace Baseliner
#endif // BASELINER_STATS_TYPE_HPP