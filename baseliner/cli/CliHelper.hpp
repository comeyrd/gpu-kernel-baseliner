#ifndef BASELINER_CLI_CLIHELPER_HPP
#define BASELINER_CLI_CLIHELPER_HPP
#include <algorithm>
#include <baseliner/core/Conversions.hpp>
#include <baseliner/core/IPrinter.hpp>
#include <baseliner/orchestrator/Plan.hpp>
#include <cstddef>
#include <iomanip>
#include <string>
#include <vector>
namespace Baseliner::Cli {
  constexpr size_t MIN_COL_SIZE = 12;
  struct Cell {
    std::string id;
    std::string name;
    size_t width;
    // Add this so std::sort works
    auto operator<(const Cell &other) const -> bool {
      return id < other.id;
    }
  };
  using Row = std::vector<Cell>;

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
  inline auto get_headers(const SingleRunReport &report) -> Row {
    std::vector<Cell> headers;

    if (report.m_sweep_point.has_value()) {
      for (const auto &[interface_name, options] : report.m_sweep_point.value()) {
        for (const auto &[opt_name, opt] : options) {
          std::string name = interface_name + "." + opt_name;
          size_t size = opt.m_value.size();
          size = std::max(size, MIN_COL_SIZE);
          size = std::max(name.size(), size);
          headers.push_back({name, name, size});
        }
      }
      std::sort(headers.begin(), headers.end()); // Consistency
    }
    for (const auto &metric : report.m_measurements) {
      if (!std::visit(IsVectorVisitor{}, metric.m_data)) {
        std::string name = metric.m_name + " (" + metric.m_unit + ")";
        std::string id = metric.m_name;
        auto variant = Conversion::baseliner_to_string(metric.m_data);
        size_t size = std::get<std::string>(variant).size();
        size = std::max(size, MIN_COL_SIZE);
        size = std::max(name.size(), size);
        headers.push_back({id, name, size});
      }
    }
    return headers;
  }
  inline void print_headers(const Row &headers) {
    for (const auto &cell : headers) {
      std::cout << std::left << std::setw(cell.width) << cell.name << "|";
    }
    std::cout << "\n";

    // 2. Print Separator Line (the "old" +-----+ logic)
    for (const auto &cell : headers) {
      std::cout << std::setfill('-') << std::setw(cell.width) << "" << "+";
    }
    std::cout << "\n" << std::setfill(' '); // Always reset setfill
  }

  inline void print_row(const Row &headers, const SingleRunReport &report) {
    for (const auto &col : headers) {
      std::string value;

      // Logic: If the ID contains a '.', it's likely an Option (Interface.Name)
      auto dot_pos = col.id.find('.');
      if (dot_pos != std::string::npos && report.m_sweep_point.has_value()) {
        std::string interface_name = col.id.substr(0, dot_pos);
        std::string opt_name = col.id.substr(dot_pos + 1);

        // Access the specific option value
        const auto &sweep = report.m_sweep_point.value();
        if (auto it_int = sweep.find(interface_name); it_int != sweep.end()) {
          if (auto it_opt = it_int->second.find(opt_name); it_opt != it_int->second.end()) {
            value = it_opt->second.m_value;
          }
        }
      } else {
        // Otherwise, it's a Metric
        for (const auto &metric : report.m_measurements) {
          if (metric.m_name == col.id) {
            auto variant = Conversion::baseliner_to_string(metric.m_data);
            value = std::get<std::string>(variant);
          }
        }
      }
      std::cout << std::left << std::fixed << std::setw(col.width) << value << "|";
    }
    std::cout << "\n";
  }

  class CliPrinter : public IBenchmarkPrinter {
  public:
    ~CliPrinter() = default;
    void consume_single_run_report(const SingleRunReport &report) override {
      if (first) {
        row = get_headers(report);
        print_headers(row);
        first = false;
      }
      print_row(row, report);
    };
    void print_campaign_plan(const CampaignPlan &campaign_plan) {
      std::cout << "Running Campaign " << campaign_plan.name << ", Recipe : " << campaign_plan.recipe_name << "\n";
    }
    void print_benchmark_plan(const BenchmarkPlan &bench_plan) {
      std::cout << "Case : " << bench_plan.m_case.m_impl << " (" << bench_plan.m_case.m_preset << ")" << "\n";
      std::cout << "Backend : " << bench_plan.m_backend.m_impl << " (" << bench_plan.m_backend.m_preset << ")" << "\n";
    };

  private:
    Row row;
    bool first = true;
  };

} // namespace Baseliner::Cli
#endif // BASELINER_CLI_HELPER