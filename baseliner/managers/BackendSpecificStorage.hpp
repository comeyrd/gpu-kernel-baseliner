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
    CaseStorage<BackendT>() = default;

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
    BenchmarkStorage<BackendT>() = default;

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
    [[nodiscard]] auto at(const std::string &name) const -> std::function<std::shared_ptr<Benchmark<BackendT>>()> {
      return m_stats_map.at(name);
    };
    BackendStatsStorage<BackendT>() = default;

  private:
    std::unordered_map<std::string, std::function<void(std::shared_ptr<Stats::StatsEngine>)>> m_stats_map;
  };
} // namespace Baseliner
#endif // BASELINER_BACKEND_SPECIFIC_STORAGE_HPP