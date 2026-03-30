#ifndef BASELINER_BUILDER_HPP
#define BASELINER_BUILDER_HPP
#include <baseliner/core/Benchmark.hpp>
#include <baseliner/orchestrator/Plan.hpp>
#include <baseliner/registry/StorageManager.hpp>

namespace Baseliner::Builder {

  auto build(const Plan &plan, const StorageManager *registry) -> IBenchmarkFactory;

} // namespace Baseliner::Builder

#endif // BASELINER_BUILDER_HPP