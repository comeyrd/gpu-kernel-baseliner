#ifndef BASELINER_ORCHESTRATOR_BUILDER_HPP
#define BASELINER_ORCHESTRATOR_BUILDER_HPP
#include <baseliner/core/Benchmark.hpp>
#include <baseliner/orchestrator/Plan.hpp>
#include <baseliner/registry/StorageManager.hpp>

namespace Baseliner::Builder {

  auto build(const BenchmarkPlan &plan, const StorageManager *registry) -> IBenchmarkFactory;

} // namespace Baseliner::Builder

#endif // BASELINER_BUILDER_HPP