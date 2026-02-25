#ifndef BASELINER_SUITE_HPP
#define BASELINER_SUITE_HPP
#include <baseliner/Axe.hpp>
#include <baseliner/Benchmark.hpp>
#include <baseliner/Conversions.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Result.hpp>
#include <memory>
#include <string>
#include <vector>
namespace Baseliner {

  class ISuite : public LazyOption {
  public:
    virtual auto name() -> std::string = 0;
    void set_benchmark(const std::function<std::shared_ptr<IBenchmark>()> &benchmark_builder) {
      m_benchmark = benchmark_builder();
    }
    [[nodiscard]] virtual auto run_all() -> std::vector<Result> = 0;

  private:
    std::shared_ptr<IBenchmark> m_benchmark;
  };
} // namespace Baseliner
#endif // BASELINER_SUITE_HPP