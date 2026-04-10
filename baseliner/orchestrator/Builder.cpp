#include <baseliner/core/Benchmark.hpp>
#include <baseliner/orchestrator/Builder.hpp>
#include <baseliner/registry/StorageManager.hpp>
#include <functional>
namespace Baseliner::Builder {

  auto build(const BenchmarkPlan &plan, const StorageManager *storage_manager) -> IBenchmarkFactory {

    IBenchmarkFactory benchmark_factory =
        storage_manager->get_benchmark_workload_factory(plan.backend.impl, plan.benchmark.impl, plan.workload.impl);

    StoppingCriterionFactory stopping_factory = storage_manager->get_stopping_criterion_factory(plan.stopping.impl);
    StatsFactory combined_stats = storage_manager->get_combined_stats_factories(plan.backend.impl, plan.stats.stats);

    stopping_factory = inject_option(stopping_factory, plan.stopping.options);
    benchmark_factory =
        inject_option(benchmark_factory, plan.benchmark.options, plan.workload.options, plan.stats.options);

    IBenchmarkFactory final_factory = [benchmark_factory, stopping_factory, plan]() -> std::shared_ptr<IBenchmark> {
      std::shared_ptr<IBenchmark> bench = benchmark_factory();
      bench->set_stopping_criterion(stopping_factory);
      bench->set_sweep_spec(plan.sweep);
      bench->set_backend_options(plan.backend.options);
      return bench;
    };
    return final_factory;
  };

} // namespace Baseliner::Builder