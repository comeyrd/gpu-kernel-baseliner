#ifndef BASELINER_SUITE_HPP
#define BASELINER_SUITE_HPP
#include <baseliner/Axe.hpp>
#include <baseliner/Benchmark.hpp>
#include <baseliner/Conversions.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/Task.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>
namespace Baseliner {

  class SingleAxeSuite : public IOption, public ITask {
  public:
    SingleAxeSuite(std::shared_ptr<IBenchmark> benchmark, Axe Axe)
        : m_benchmark(std::move(benchmark)),
          m_axe(std::move(Axe)) {};
    std::string name() override {
      return "SingleAxeSuite";
    }
    auto run_all() -> std::vector<Result> override {
      std::vector<Result> results_v{};
      const OptionsMap baseMap;
      m_benchmark->gather_options();
      OptionsMap tempMap;
      for (const std::string &axe_val : m_axe.get_values()) {
        tempMap = baseMap;
        tempMap[m_axe.get_interface_name()][m_axe.get_option_name()].m_value = axe_val;
        m_benchmark->propagate_options(tempMap);
        const Result result = m_benchmark->run();
        results_v.push_back(result);
      }
      return results_v;
    };

    std::string print_console(const std::vector<Result> &results) override {
      if (results.empty())
        return "Task " + name() + "\nNo results.\n";

      std::stringstream sstream;
      sstream << "Task " << name() << "\n";
      sstream << m_benchmark->name() << "\n";
      // Define standard widths to ensure alignment
      const int AXE_WIDTH = 12;
      const int COL_WIDTH = 20;
      sstream << "Comparing : " << m_axe.get_interface_name() << "." << m_axe.get_option_name() << "\n\n";

      sstream << "Metric : " << "\n";

      for (auto &metric : results[0].get_v_metrics()) {
        sstream << std::left << std::setw(12) << metric.m_name << " : ";
        for (auto &stat : metric.m_v_stats) {
          bool is_vector = std::holds_alternative<std::vector<float_milliseconds>>(stat.m_data) ||
                           std::holds_alternative<std::vector<int64_t>>(stat.m_data) ||
                           std::holds_alternative<std::vector<std::string>>(stat.m_data) ||
                           std::holds_alternative<std::vector<float>>(stat.m_data);

          if (!is_vector) {
            sstream << stat.m_name << " , ";
          }
        }
        sstream << "\n";
      }
      sstream << "\n";

      sstream << std::left << std::setw(AXE_WIDTH) << "Axe" << " | ";

      struct StatRef {
        std::string metric_name;
        std::string stat_name;
      };
      std::vector<StatRef> active_stats;

      for (auto &metric : results[0].get_v_metrics()) {
        for (auto &stat : metric.m_v_stats) {
          bool is_vector = std::holds_alternative<std::vector<float_milliseconds>>(stat.m_data) ||
                           std::holds_alternative<std::vector<int64_t>>(stat.m_data) ||
                           std::holds_alternative<std::vector<std::string>>(stat.m_data) ||
                           std::holds_alternative<std::vector<float>>(stat.m_data);

          if (!is_vector) {
            active_stats.push_back({metric.m_name, stat.m_name});
            std::string header_label = stat.m_name + " " + metric.m_unit;
            sstream << std::left << std::setw(COL_WIDTH) << header_label << " | ";
          }
        }
      }
      sstream << "\n";

      sstream << std::string(AXE_WIDTH, '-') << "-|-" << std::string(active_stats.size() * (COL_WIDTH + 3), '-')
              << "\n";

      for (auto &item : results) {
        const OptionsMap omap = item.get_map();

        std::string axe_val = omap.at(m_axe.get_interface_name()).at(m_axe.get_option_name()).m_value;
        sstream << std::left << std::setw(AXE_WIDTH) << axe_val << " | ";

        for (const auto &ref : active_stats) {
          std::string val_str = "N/A";

          for (auto &metric : item.get_v_metrics()) {
            if (metric.m_name == ref.metric_name) {
              for (auto &stat : metric.m_v_stats) {
                if (stat.m_name == ref.stat_name) {
                  val_str = Conversion::baseliner_to_string(stat.m_data);
                  break;
                }
              }
            }
          }
          sstream << std::left << std::setw(COL_WIDTH) << val_str << " | ";
        }
        sstream << "\n";
      }

      return sstream.str();
    }

  protected:
    void register_options_dependencies() override {
      register_consumer(*m_benchmark);
      register_consumer(m_axe);
    };
    void register_options() override {
    }

  private:
    std::shared_ptr<IBenchmark> m_benchmark;
    Axe m_axe;
  };

} // namespace Baseliner
#endif // BASELINER_SUITE_HPP