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

#ifdef __linux__
#include <sys/ioctl.h>
#include <unistd.h>
inline auto get_max_size() -> size_t {
  struct winsize w;
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) {
    return w.ws_col;
  }
  return 180;
}
#else
inline auto get_max_size() -> size_t {
  return 180;
}
#endif

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
  inline auto get_headers(const RunReport &report) -> Row {
    std::vector<Cell> headers;

    if (report.sweep_point.has_value()) {
      for (const auto &[interface_name, options] : report.sweep_point.value()) {
        for (const auto &[opt_name, opt] : options) {
          std::string name = interface_name + "." + opt_name;
          size_t size = opt.value.size();
          size = std::max(size, MIN_COL_SIZE);
          size = std::max(name.size(), size);
          headers.push_back({name, name, size});
        }
      }
      std::sort(headers.begin(), headers.end()); // Consistency
    }
    for (const auto &metric : report.measurements) {
      if (!std::visit(IsVectorVisitor{}, metric.data)) {
        std::string name = metric.name + " (" + metric.unit + ")";
        std::string id = metric.name;
        auto variant = Conversion::baseliner_to_string(metric.data);
        size_t size = std::get<std::string>(variant).size();
        size = std::max(size, MIN_COL_SIZE);
        size = std::max(name.size(), size);
        headers.push_back({id, name, size});
      }
    }
    size_t curr_sz = 0;
    std::vector<Cell> full_header;
    full_header.reserve(headers.size());
    size_t max_size = get_max_size();
    for (auto &cell : headers) {
      if ((curr_sz + cell.width + 1) > max_size) {
        break;
      }
      curr_sz += cell.width + 1;
      full_header.push_back(cell);
    }
    full_header.shrink_to_fit();
    if (full_header.size() != headers.size()) {
      std::string missing_head{};
      bool first{true};
      for (size_t pos = full_header.size() - 1; pos < headers.size(); pos++) {
        if (!first) {
          missing_head += ", ";
        }
        missing_head += headers[pos].name;
        first = false;
      }
      std::cout << "Warning : Terminal output is truncated due to terminal size missing " << missing_head
                << " | full output in report file\n";
    }
    return full_header;
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

  inline void print_row(const Row &headers, const RunReport &report) {
    for (const auto &col : headers) {
      std::string value;

      // Logic: If the ID contains a '.', it's likely an Option (Interface.Name)
      auto dot_pos = col.id.find('.');
      if (dot_pos != std::string::npos && report.sweep_point.has_value()) {
        std::string interface_name = col.id.substr(0, dot_pos);
        std::string opt_name = col.id.substr(dot_pos + 1);

        // Access the specific option value
        const auto &sweep = report.sweep_point.value();
        if (auto it_int = sweep.find(interface_name); it_int != sweep.end()) {
          if (auto it_opt = it_int->second.find(opt_name); it_opt != it_int->second.end()) {
            value = it_opt->second.value;
          }
        }
      } else {
        // Otherwise, it's a Metric
        for (const auto &metric : report.measurements) {
          if (metric.name == col.id) {
            auto variant = Conversion::baseliner_to_string(metric.data);
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
    void consume_single_run_report(const RunReport &report) override {
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
      std::cout << "Workload : " << bench_plan.workload.impl << " (" << bench_plan.workload.preset << ")" << "\n";
      std::cout << "Backend : " << bench_plan.backend.impl << " (" << bench_plan.backend.preset << ")" << "\n";
    };

  private:
    Row row;
    bool first = true;
  };

} // namespace Baseliner::Cli
#endif // BASELINER_CLI_HELPER