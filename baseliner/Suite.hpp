#ifndef BASELINER_SUITE_HPP
#define BASELINER_SUITE_HPP
#include <baseliner/Axe.hpp>
#include <baseliner/Benchmark.hpp>
#include <baseliner/Conversions.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Result.hpp>

#include <iomanip>
#include <memory>
#include <string>
#include <utility>
#include <vector>
namespace Baseliner {

  class ISuite : public LazyOption {
  public:
    virtual auto name() -> std::string = 0;
    void set_benchmark(const std::function<std::shared_ptr<IBenchmark>()> &benchmark_builder) {
      m_benchmark = benchmark_builder();
    }

  private:
    std::shared_ptr<IBenchmark> m_benchmark;
  };

  class SingleAxeSuite : ISuite {
  public:
    SingleAxeSuite() = default;
    auto name() -> std::string override {
      return "SingleAxeSuite";
    }
    [[nodiscard]] auto run_all() -> std::vector<Result> {
      if (m_axe.get_interface_name().empty()) {
        throw std::runtime_error("Error : Suite launched without axes");
      }
      std::vector<Result> results_v{};
      const OptionsMap baseMap = m_benchmark->gather_options();
      OptionsMap tempMap;
      // Checks that the Axe interface name and option exist
      if (baseMap.find(m_axe.get_interface_name()) == baseMap.end()) {
        throw std::runtime_error("Axe error : The interface " + m_axe.get_interface_name() + " does not exist");
      }
      if (baseMap.at(m_axe.get_interface_name()).find(m_axe.get_option_name()) ==
          baseMap.at(m_axe.get_interface_name()).end()) {
        throw std::runtime_error("Axe error : No option " + m_axe.get_option_name() + " in interface " +
                                 m_axe.get_interface_name());
      }
      for (const std::string &axe_val : m_axe.get_values()) {
        tempMap = baseMap;
        tempMap[m_axe.get_interface_name()][m_axe.get_option_name()].m_value = axe_val;
        m_benchmark->propagate_options(tempMap);
        const Result result = m_benchmark->run();
        results_v.push_back(result);
        print_result(result);
      }
      return results_v;
    };
    void print_result(const Result &result) {
      if (m_axe.get_interface_name().empty()) {
        throw std::runtime_error("Error : Suite print result without axes");
      }
      std::stringstream sstream;
      const auto &metrics = result.get_v_metrics();
      const OptionsMap &omap = result.get_map();

      // 1. Get Axis Value
      std::string axe_val = omap.at(m_axe.get_interface_name()).at(m_axe.get_option_name()).m_value;

      if (m_first_result) {
        m_first_result = false;

        // --- CALCULATION PHASE ---
        const int MIN_WIDTH = 12;
        const int PADDING = 2;

        // Calculate Axis Width (Left Column)
        int axe_h_len = static_cast<int>(m_axe.get_option_name().length());
        int axe_v_len = static_cast<int>(axe_val.length());
        int final_axe_width = std::max({MIN_WIDTH, axe_h_len, axe_v_len}) + PADDING;
        m_col_widths.push_back(final_axe_width);

        // Header for Axis
        std::stringstream header_line;
        header_line << std::left << std::setw(final_axe_width) << m_axe.get_option_name();

        // Calculate Metric Widths (Data Columns)
        for (const auto &metric : metrics) {
          // Skip vectors (non-scalar data)
          bool is_vector = std::holds_alternative<std::vector<float_milliseconds>>(metric.m_data) ||
                           std::holds_alternative<std::vector<int64_t>>(metric.m_data) ||
                           std::holds_alternative<std::vector<std::string>>(metric.m_data) ||
                           std::holds_alternative<std::vector<float>>(metric.m_data);

          if (!is_vector) {
            std::string header_text = metric.m_name + (metric.m_unit.empty() ? "" : " (" + metric.m_unit + ")");
            std::string sample_val = Conversion::baseliner_to_string(metric.m_data) + " " + metric.m_unit;

            // Right-aligned columns need to fit the widest element
            int col_w = std::max({MIN_WIDTH, (int)header_text.length(), (int)sample_val.length()}) + PADDING;

            m_col_widths.push_back(col_w);
            m_metric_names.push_back(metric.m_name);

            // Metrics are RIGHT aligned
            header_line << std::right << std::setw(col_w) << header_text;
          }
        }

        // --- PRINTING HEADER ---
        int total_width = 0;
        for (int w : m_col_widths)
          total_width += w;

        std::cout << "Task " << name() << " | " << m_benchmark->name() << "\n";
        std::cout << header_line.str() << "\n";
        std::cout << std::string(total_width, '-') << "\n";
      }

      // --- 2. PRINT DATA ROW ---
      // Axis: Left Aligned
      sstream << std::left << std::setw(m_col_widths[0]) << axe_val;

      // Metrics: Right Aligned
      size_t width_idx = 1;
      for (const auto &target_name : m_metric_names) {
        auto it = std::find_if(metrics.begin(), metrics.end(), [&](const auto &m) { return m.m_name == target_name; });

        if (it != metrics.end() && width_idx < m_col_widths.size()) {
          std::string val_str = Conversion::baseliner_to_string(it->m_data);
          if (!it->m_unit.empty())
            val_str += " " + it->m_unit;

          sstream << std::right << std::setw(m_col_widths[width_idx]) << val_str;
          width_idx++;
        }
      }

      sstream << "\n";
      std::cout << sstream.str();
    }

  private:
    std::shared_ptr<IBenchmark> m_benchmark;
    SingleAxe m_axe;
    bool m_first_result = true;
    std::vector<int> m_col_widths;
    std::vector<std::string> m_metric_names;
  };

} // namespace Baseliner
#endif // BASELINER_SUITE_HPP