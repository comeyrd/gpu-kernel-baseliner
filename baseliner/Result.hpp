#ifndef BASELINER_RESULT_HPP
#define BASELINER_RESULT_HPP
#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>

#include <string>
namespace Baseliner {
  struct Result {
    const OptionsMap m_map;
    const std::string m_kernel_name;
    const std::string m_git_version;
    const std::string m_execution_uid;
    const std::string m_date_time;
    std::vector<Metric> m_v_metrics;
    explicit Result() {};
    explicit Result(const OptionsMap &omap, std::string kernel_name, std::string git_version)
        : m_map(omap),
          m_kernel_name(kernel_name),
          m_git_version(git_version),
          m_execution_uid(generate_uid()),
          m_date_time(current_time_string()),
          m_v_metrics() {};
    void push_back_metric(Metric &metric) {
      m_v_metrics.push_back(metric);
    };
    void push_back_metrics(std::vector<Metric> &metrics) {
      for (auto &m : metrics) {
        m_v_metrics.push_back(std::move(m));
      }
    };

  private:
    static std::string current_time_string();
    static std::string generate_uid();
  };

} // namespace Baseliner

#endif // BASELINER_RESULT_HPP