#ifndef BASELINER_BUILDER_HPP
#define BASELINER_BUILDER_HPP
#include <baseliner/Benchmark.hpp>
#include <baseliner/managers/StorageManager.hpp>

namespace Baseliner::Builder {

  auto build(const Plan &plan, const StorageManager *registry) -> IBenchmarkFactory;

} // namespace Baseliner::Builder

#endif // BASELINER_BUILDER_HPP