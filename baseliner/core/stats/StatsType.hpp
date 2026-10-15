#ifndef BASELINER_CORE_STATS_STATSTYPE_HPP
#define BASELINER_CORE_STATS_STATSTYPE_HPP
#include <baseliner/cli/Serializer.hpp>
#include <baseliner/core/Conversions.hpp>

namespace Baseliner {
  enum class MetricGranularity : uint8_t {
    EVERY_ELEMENT,
    EVERY_BATCH,
    ON_DEMAND,
    ONCE,
  };

  enum class MetricVerbosity : uint8_t {
    CORE = 0,            // Level 0: Core scalars (reps, batch count, mean, median, cv)
    SCALARS = 1,         // Level 1: All scalars
    BATCH_VECTORS = 2,   // Level 2: All scalars + batch vectors
    ELEMENT_VECTORS = 3, // Level 3: All + every-element vectors
  };

} // namespace Baseliner
#endif // BASELINER_STATS_TYPE_HPP