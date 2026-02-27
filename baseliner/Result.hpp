#ifndef BASELINER_RESULT_HPP
#define BASELINER_RESULT_HPP
#include "baseliner/ConfigFile.hpp"
#include <baseliner/GIT_VERSION.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Version.hpp>
#include <iterator>
#include <string>
#include <utility>
#include <variant>
#include <vector>
namespace Baseliner {

  auto generate_uid() -> std::string;
  auto current_time_string() -> std::string;
  struct BenchmarkResult {
    std::variant<OptionsMap, std::monostate> m_options;
    std::vector<Metric> m_v_metrics;
  };
  inline static auto build_benchmark_result(std::vector<Metric> &metrics_vector) -> BenchmarkResult {
    return BenchmarkResult{std::monostate(), std::move(metrics_vector)};
  };
  struct RunResult {
    Recipe m_recipe;
    std::string m_run_uuid;
    std::vector<BenchmarkResult> m_results;
  };
  inline static auto build_run_result(std::vector<BenchmarkResult> &results) -> RunResult {
    return RunResult{{}, generate_uid(), std::move(results)};
  }
  struct Result {
    std::string m_baseliner_version;
    std::string m_git_version;
    std::string m_date_time;

    std::vector<PresetDefinition> m_presets;
    std::vector<RunResult> m_runs;
  };
  inline static auto build_result() -> Result {
    return Result{std::string(Version::string), BASELINER_GIT_VERSION, current_time_string(), {}, {}};
  }
  class Result2 {
  public:
    explicit Result2(OptionsMap omap, std::string kernel_name, bool valid)
        : m_map(std::move(omap)),
          m_kernel_name(std::move(kernel_name)),
          m_git_version(BASELINER_GIT_VERSION),
          m_execution_uid(generate_uid()),
          m_date_time(current_time_string()),
          m_baseliner_version(Version::string),
          m_valid(valid) {};
    void push_back_metric(Metric &metric) {
      m_v_metrics.push_back(metric);
    };
    void push_back_metrics(std::vector<Metric> &metrics) {
      m_v_metrics.insert(m_v_metrics.end(), std::make_move_iterator(metrics.begin()),
                         std::make_move_iterator(metrics.end()));
    }

    auto get_map() const -> const OptionsMap & {
      return m_map;
    }
    auto get_kernel_name() const -> const std::string & {
      return m_kernel_name;
    }
    auto get_git_version() const -> const std::string & {
      return m_git_version;
    }
    auto get_basliner_version() const -> const std::string & {
      return m_baseliner_version;
    }
    auto get_execution_uid() const -> const std::string & {
      return m_execution_uid;
    }
    auto get_date_time() const -> const std::string & {
      return m_date_time;
    }
    auto get_v_metrics() const -> const std::vector<Metric> & {
      return m_v_metrics;
    }

  private:
    OptionsMap m_map;
    std::string m_kernel_name;
    std::string m_git_version;
    std::string m_execution_uid;
    std::string m_date_time;
    std::string m_baseliner_version;
    std::vector<Metric> m_v_metrics;
    bool m_valid{};
    explicit Result2() = default;
    static auto current_time_string() -> std::string;
    static auto generate_uid() -> std::string;
  };

} // namespace Baseliner

#endif // BASELINER_RESULT_HPP