#include <baseliner/Benchmark.hpp>
#include <baseliner/Builder.hpp>
#include <baseliner/managers/StorageManager.hpp>
#include <functional>
namespace Baseliner::Builder {

  auto build(const Plan &plan, const StorageManager &registry) -> Execution {

    IBenchmarkFactory benchmark_factory =
        registry.get_benchmark_case_factory(plan.m_backend.m_impl, plan.m_benchmark.m_impl, plan.m_case.m_impl);

    StoppingCriterionFactory stopping_factory = registry.get_stopping_criterion_factory(plan.m_stopping.m_impl);
    StatsFactory combined_stats = registry.get_combined_stats_factories(plan.m_backend.m_impl, plan.m_stats.m_stats);

    stopping_factory = inject_option(stopping_factory, plan.m_stopping.m_options);
    benchmark_factory =
        inject_option(benchmark_factory, plan.m_benchmark.m_options, plan.m_case.m_options, plan.m_stats.m_options);

    Execution built_execution;
    built_execution.m_benchmark_factory = [benchmark_factory, stopping_factory]() -> std::shared_ptr<IBenchmark> {
      std::shared_ptr<IBenchmark> bench = benchmark_factory();
      bench->set_stopping_criterion(stopping_factory);
    };
    built_execution.m_backend_setup = registry.get_backend_setup(plan.m_backend.m_impl, plan.m_backend.m_options);

    return built_execution;
  };

} // namespace Baseliner::Builder