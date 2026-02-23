#ifndef BASELINER_RESULT_HPP
#define BASELINER_RESULT_HPP
#include <baseliner/GIT_VERSION.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Version.hpp>
#include <iterator>
#include <string>
#include <utility>
#include <vector>
namespace Baseliner {
  class Result {
  public:
    explicit Result(OptionsMap omap, std::string kernel_name, bool valid)
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
    explicit Result() = default;
    static auto current_time_string() -> std::string;
    static auto generate_uid() -> std::string;
  };

} // namespace Baseliner

#endif // BASELINER_RESULT_HPP