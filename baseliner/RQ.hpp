#ifndef BASELINER_RQ_HPP
#define BASELINER_RQ_HPP
#include <baseliner/Axe.hpp>
#include <baseliner/Benchmark.hpp>
#include <baseliner/managers/BenchmarkCaseManager.hpp>
#include <baseliner/managers/SuiteManager.hpp>
namespace Baseliner {
  template <typename BackendT>
  auto build_RQ_Benchmark() -> std::shared_ptr<Benchmark<BackendT>> {
    auto bench = std::make_shared<Benchmark<BackendT>>();
    bench->set_flush_l2(true);
    bench->set_block(false);
    return bench;
  }
} // namespace Baseliner
#endif // BASELINER_RQ_HPP