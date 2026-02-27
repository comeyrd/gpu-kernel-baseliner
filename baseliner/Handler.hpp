#ifndef BASELINER_HANDLER_HPP
#define BASELINER_HANDLER_HPP
#include "baseliner/Benchmark.hpp"
#include "baseliner/Result.hpp"
#include "baseliner/Suite.hpp"
#include "baseliner/managers/Manager.hpp"
#include <functional>
#include <variant>
namespace Baseliner {

  class Handler {
  public:
    Handler() = default;

    [[nodiscard]] auto run_recipes(std::vector<Recipe> &recipes) -> Result {
      Result result = build_result();
      PresetSet set;
      for (auto &recipe : recipes) {
        result.m_runs.push_back(run_recipe(recipe, set));
      }
      std::vector<PresetDefinition> preset_defs;
      preset_defs.insert(preset_defs.end(), set.begin(), set.end());
      result.m_presets = preset_defs;
      return result;
    }

  private:
    [[nodiscard]] auto run_recipe(Recipe &recipe, PresetSet &set) -> RunResult {
      auto bench_or_suite = m_manager->build_recipe(recipe, set);
      RunResult result;
      if (std::holds_alternative<std::function<std::shared_ptr<IBenchmark>()>>(bench_or_suite)) {
        result = run_benchmark(std::get<std::function<std::shared_ptr<IBenchmark>()>>(bench_or_suite));
      } else {
        result = run_suite(std::get<std::function<std::shared_ptr<ISuite>()>>(bench_or_suite));
      }
      result.m_recipe = recipe;
      return result;
    };

    [[nodiscard]] static auto run_benchmark(const std::function<std::shared_ptr<IBenchmark>()> &bench) -> RunResult {
      BenchmarkResult bench_result = bench()->run();
      bench_result.m_options = std::monostate();
      std::vector<BenchmarkResult> results{bench_result};
      return build_run_result(results);
    }
    static auto run_suite(const std::function<std::shared_ptr<ISuite>()> &suite) -> RunResult {
      return suite()->run_all();
    }
    Manager *m_manager = Manager::instance();
  };

} // namespace Baseliner
#endif // BASELINER_HANDLER_HPP