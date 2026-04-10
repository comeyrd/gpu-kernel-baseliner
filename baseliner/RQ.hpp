#ifndef BASELINER_RQ_HPP
#define BASELINER_RQ_HPP
#include <baseliner/core/AxeSweeping.hpp>
#include <baseliner/core/Benchmark.hpp>
#include <baseliner/orchestrator/Protocol.hpp>

namespace Baseliner {

  enum class RQSize {
    Small,
    Medium,
    Large
  };

  auto rq_recipes(RQSize size) -> std::unordered_map<std::string, Recipe>;
  auto rq_protocol(RQSize size, std::vector<RecipeComponent> &workloads, std::vector<RecipeComponent> &backends)
      -> Protocol;
} // namespace Baseliner
#endif // BASELINER_RQ_HPP