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

  using BackendSetup = std::function<void()>;

  using ComponentList = std::vector<std::pair<std::string, ComponentType>>;
  using ComponentPresetList = std::vector<std::pair<std::string, ComponentPreset>>;
  using StatsPresetList = std::vector<std::pair<std::string, StatsPreset>>;
} // namespace Baseliner
#endif // BASELINER_FACTORIES_HPP