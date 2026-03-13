#ifndef BASELINER_BUILDER_HPP
#define BASELINER_BUILDER_HPP
#include <baseliner/Benchmark.hpp>
#include <baseliner/managers/StorageManager.hpp>
#include <functional>
namespace Baseliner::Builder {

  struct Execution {
    std::function<std::shared_ptr<IBenchmark>()> m_benchmark_factory;
    std::function<void()> m_backend_setup;
  };

  auto build(const Plan &plan, const StorageManager &registry) -> Execution;

} // namespace Baseliner::Builder
#endif // BASELINER_BUILDER_HPP