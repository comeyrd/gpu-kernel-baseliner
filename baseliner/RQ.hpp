#ifndef BASELINER_RQ_HPP
#define BASELINER_RQ_HPP
#include <baseliner/AxeSweeping.hpp>
#include <baseliner/Benchmark.hpp>

namespace Baseliner {

  enum class RQSize {
    Small,
    Medium,
    Large
  };

  auto rq_recipes(RQSize size) -> std::unordered_map<std::string, Recipe>;
  auto rq_protocol(RQSize size, std::vector<RecipeComponent> &cases, std::vector<RecipeComponent> &backends)
      -> Protocol;
} // namespace Baseliner
#endif // BASELINER_RQ_HPP