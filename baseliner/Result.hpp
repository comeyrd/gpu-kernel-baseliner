#ifndef BASELINER_RESULT_HPP
#define BASELINER_RESULT_HPP
#include <baseliner/ConfigFile.hpp>
#include <baseliner/GIT_VERSION.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Version.hpp>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <type_traits>
#include <unordered_map>
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

  template <typename T>
  struct is_vector : std::false_type {};

  template <typename T, typename Alloc>
  struct is_vector<std::vector<T, Alloc>> : std::true_type {};
  struct IsVectorVisitor {
    template <typename T>
    bool operator()(const T &) const {
      return is_vector<T>::value;
    }
  };

  static void print_benchmark_result(std::ostream &oss, const BenchmarkResult &result, bool first) {

    if (std::holds_alternative<std::monostate>(result.m_options)) {
      const int col_width = 24;
      for (const auto &metric : result.m_v_metrics) {
        if (!std::visit(IsVectorVisitor{}, metric.m_data)) {
          oss << std::left << std::setw(col_width) << (metric.m_name + " (" + metric.m_unit + ") : ");
          oss << Conversion::baseliner_to_string(metric.m_data);
          oss << "\n";
        }
      }
      return;
    }
    const auto &unordered_options = std::get<OptionsMap>(result.m_options);
    std::map<std::string, Option> ordered_options;
    for (const auto &[interface_name, interface_options] : unordered_options) {
      for (const auto &[option_name, option] : interface_options) {
        ordered_options[interface_name + "." + option_name] = option;
      }
    }
    int min_col_size = 12;
    std::vector<int> colsizes;
    colsizes.resize(ordered_options.size(), min_col_size);
    {
      int _count = 0;
      for (const auto &[name, option] : ordered_options) {
        int temp_size = name.size();
        if (temp_size > min_col_size) {
          colsizes[_count] = temp_size;
        }
        _count++;
      };
    }
    std::stringstream first_ss;
    {
      first_ss << std::left;
      int count = 0;
      for (const auto &[name, tuple] : ordered_options) {
        first_ss << std::setw(colsizes[count]) << name << " | ";
        count++;
      }
      for (const auto &metric : result.m_v_metrics) {
        if (!std::visit(IsVectorVisitor{}, metric.m_data)) {
          std::string inside_str = metric.m_name + " (" + metric.m_unit + ") ";
          int temp_size = inside_str.size();
          if (temp_size > min_col_size) {
            colsizes.push_back(temp_size);
          } else {
            colsizes.push_back(min_col_size);
          }
          first_ss << std::setw(colsizes[count]) << inside_str << " | ";
          count++;
        }
      }
    }
    if (first) {
      oss << first_ss.str();
      oss << "\n";
    }

    // Affichage des valeurs de la ligne courante

    int count = 0;
    for (const auto &[interface_name, interface_options] : ordered_options) {
      oss << std::left << std::fixed << std::setprecision(6) << std::setw(colsizes[count]) << interface_options.m_value
          << " | ";
      count++;
    }

    for (const auto &metric : result.m_v_metrics) {
      if (!std::visit(IsVectorVisitor{}, metric.m_data)) {
        oss << std::setw(colsizes[count]) << Conversion::baseliner_to_string(metric.m_data) << " | ";
        count++;
      }
    }
    oss << "\n";
  }
} // namespace Baseliner

#endif // BASELINER_RESULT_HPP