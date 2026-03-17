#ifndef BASELINER_BUILDER_HPP
#define BASELINER_BUILDER_HPP
#include <baseliner/Benchmark.hpp>
#include <baseliner/managers/StorageManager.hpp>
#include <functional>
namespace Baseliner {
  namespace Builder {

    auto build(const Plan &plan, const StorageManager &registry) -> IBenchmarkFactory;

  } // namespace Builder
} // namespace Baseliner
#endif // BASELINER_BUILDER_HPP