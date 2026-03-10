#ifndef BASELINER_FACTORIES_HPP
#define BASELINER_FACTORIES_HPP
#include <baseliner/Benchmark.hpp>
#include <baseliner/Case.hpp>
#include <functional>
namespace Baseliner {
  // Factories
  template <typename BackendT>
  using CaseFactory = std::function<std::shared_ptr<ICase<BackendT>>()>;

  template <typename BackendT>
  using BenchmarkFactory = std::function<std::shared_ptr<Benchmark<BackendT>>()>;

  using IBenchmarkFactory = std::function<std::shared_ptr<IBenchmark>()>;
  using StoppingCriterionFactory = std::function<std::unique_ptr<StoppingCriterion>()>;
  using StatsFactory = std::function<void(std::shared_ptr<Stats::StatsEngine>)>;

} // namespace Baseliner
#endif // BASELINER_FACTORIES_HPP