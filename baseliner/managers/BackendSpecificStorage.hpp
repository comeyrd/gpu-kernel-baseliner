#ifndef BASELINER_BACKEND_SPECIFIC_STORAGE_HPP
#define BASELINER_BACKEND_SPECIFIC_STORAGE_HPP
#include <baseliner/Benchmark.hpp>
#include <baseliner/Case.hpp>
#include <baseliner/stats/StatsEngine.hpp>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace Baseliner {

  template <typename BackendT>
  class CaseStorage {
  public:
    [[nodiscard]] auto has(const std::string &name) const -> bool {
      return m_cases_map.find(name) != m_cases_map.end();
    }
    /**
     *  throws std::out_of_range If the case is not in the storage, use has_case to check beforehand.
     */
    [[nodiscard]] auto at(const std::string &name) const -> std::function<std::shared_ptr<ICase<BackendT>>()> {
      return m_cases_map.at(name);
    };
    void insert(const std::string &name, const std::function<std::shared_ptr<ICase<BackendT>>()> &case_factory,
                const std::string &backend_name) {
      if (has(name)) {
        throw std::runtime_error("Backend Specific Case : " + name + " already registered for backend" + backend_name);
      }
      m_cases_map[name] = case_factory;
    }
    CaseStorage<BackendT>() = default;
    [[nodiscard]] auto list() const -> std::vector<std::string> {
      std::vector<std::string> vecstr;
      vecstr.reserve(m_cases_map.size());
      for (const auto &[name, _] : m_cases_map) {
        vecstr.push_back(name);
      }
      return vecstr;
    }

  private:
    std::unordered_map<std::string, std::function<std::shared_ptr<ICase<BackendT>>()>> m_cases_map;
  };

  template <typename BackendT>
  class BenchmarkStorage {
  public:
    [[nodiscard]] auto has(const std::string &name) const -> bool {
      return m_benchmarks_map.find(name) != m_benchmarks_map.end();
    }
    /**
     *  throws std::out_of_range If the benchmark is not in the storage, use has to check beforehand.
     */
    [[nodiscard]] auto at(const std::string &name) const -> std::function<std::shared_ptr<Benchmark<BackendT>>()> {
      return m_benchmarks_map.at(name);
    };
    void insert(const std::string &name, const std::function<std::shared_ptr<Benchmark<BackendT>>()> &backend_factory,
                const std::string &backend_name) {
      if (has(name)) {
        throw std::runtime_error("Backend Specific Benchmark : " + name + " already registered for backend" +
                                 backend_name);
      }
      m_benchmarks_map[name] = backend_factory;
    }
    BenchmarkStorage<BackendT>() = default;

    [[nodiscard]] auto list() const -> std::vector<std::string> {
      std::vector<std::string> vecstr;
      vecstr.reserve(m_benchmarks_map.size());
      for (const auto &[name, _] : m_benchmarks_map) {
        vecstr.push_back(name);
      }
      return vecstr;
    }

  private:
    std::unordered_map<std::string, std::function<std::shared_ptr<Benchmark<BackendT>>()>> m_benchmarks_map;
  };

  template <typename BackendT>
  class BackendStatsStorage {
  public:
    [[nodiscard]] auto has(const std::string &name) const -> bool {
      return m_stats_map.find(name) != m_stats_map.end();
    }
    /**
     *  throws std::out_of_range If the benchmark is not in the storage, use has_benchmark to check beforehand.
     */
    [[nodiscard]] auto at(const std::string &name) const -> std::function<void(std::shared_ptr<Stats::StatsEngine>)> {
      return m_stats_map.at(name);
    };
    void insert(const std::string &name, const std::function<void(std::shared_ptr<Stats::StatsEngine>)> &stat_factory,
                const std::string &backend_name) {
      if (has(name)) {
        throw std::runtime_error("Backend Specific Stat : " + name + " already registered for backend" + backend_name);
      }
      m_stats_map[name] = stat_factory;
    }
    BackendStatsStorage<BackendT>() = default;

    [[nodiscard]] auto list() const -> std::vector<std::string> {
      std::vector<std::string> vecstr;
      vecstr.reserve(m_stats_map.size());
      for (const auto &[name, _] : m_stats_map) {
        vecstr.push_back(name);
      }
      return vecstr;
    }

  private:
    std::unordered_map<std::string, std::function<void(std::shared_ptr<Stats::StatsEngine>)>> m_stats_map;
  };
} // namespace Baseliner
#endif // BASELINER_BACKEND_SPECIFIC_STORAGE_HPP