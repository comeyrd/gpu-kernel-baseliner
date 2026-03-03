#ifndef BASELINER_HANDLER_HPP
#define BASELINER_HANDLER_HPP
#include <baseliner/Benchmark.hpp>
#include <baseliner/ConfigFile.hpp>
#include <baseliner/Recipe.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/State.hpp>
#include <baseliner/Suite.hpp>
#include <baseliner/managers/Manager.hpp>
#include <functional>
#include <variant>
namespace Baseliner {

  class Handler {
  public:
    Handler() = default;

    [[nodiscard]] static auto run_recipes(std::vector<Recipe> &recipes) -> Result {
      Result result = build_result();
      PresetSet set;
      for (auto &recipe : recipes) {
        if (ExecutionController::exit_requested()) {
          break;
        }
        result.m_runs.push_back(run_recipe(recipe, set));
      }
      std::vector<PresetDefinition> preset_defs;
      preset_defs.insert(preset_defs.end(), set.begin(), set.end());
      result.m_presets = preset_defs;
      return result;
    }
    [[nodiscard]] static auto run_config(Config &config) -> Result {
      Manager::instance()->add_presets(config.m_presets);
      return run_recipes(config.m_recipes);
    }
    [[nodiscard]] static auto replay_result(Result &result) -> Result {
      Manager::instance()->add_presets(result.m_presets);
      std::vector<Recipe> recipes;
      recipes.reserve(result.m_runs.size());
      for (auto &runs : result.m_runs) {
        recipes.push_back(runs.m_recipe);
      }
      return run_recipes(recipes);
    }

  private:
    [[nodiscard]] static auto run_recipe(Recipe &recipe, PresetSet &set) -> RunResult {
      auto [bench_or_suite, backend_setup] = Manager::instance()->build_recipe(recipe, set);
      RunResult result;
      backend_setup();
      print_recipe(std::cout, recipe);
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
      print_benchmark_result(std::cout, bench_result, false);
      std::vector<BenchmarkResult> results{bench_result};
      return build_run_result(results);
    }
    static auto run_suite(const std::function<std::shared_ptr<ISuite>()> &suite) -> RunResult {
      return suite()->run_all();
    }
  };

} // namespace Baseliner
#endif // BASELINER_HANDLER_HPP