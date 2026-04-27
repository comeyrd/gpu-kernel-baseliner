#ifndef BASELINER_CORE_STATS_STATSTYPE_HPP
#define BASELINER_CORE_STATS_STATSTYPE_HPP
#include <baseliner/cli/Serializer.hpp>
#include <baseliner/specs/Conversions.hpp>

namespace Baseliner {
  enum class MetricGranularity : uint8_t {
    EVERY_ELEMENT,
    EVERY_BATCH,
    ON_DEMAND,
    ONCE,
  };

} // namespace Baseliner
#endif // BASELINER_STATS_TYPE_HPP