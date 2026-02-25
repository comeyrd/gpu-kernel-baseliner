#ifndef BASELINER_GENERAL_STORAGE_HPP
#define BASELINER_GENERAL_STORAGE_HPP
#include <baseliner/Suite.hpp>
#include <baseliner/stats/StatsEngine.hpp>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
namespace Baseliner {

  class GeneralStatsStorage {
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

    GeneralStatsStorage() = default;

  private:
    std::unordered_map<std::string, std::function<void(std::shared_ptr<Stats::StatsEngine>)>> m_stats_map;
  };
  class SuiteStorage {
  public:
    [[nodiscard]] auto has(const std::string &name) const -> bool {
      return m_storage_map.find(name) != m_storage_map.end();
    }
    /**
     *  throws std::out_of_range If the benchmark is not in the storage, use has_benchmark to check beforehand.
     */
    [[nodiscard]] auto at(const std::string &name) const -> std::function<std::shared_ptr<ISuite>()> {
      return m_storage_map.at(name);
    };
    SuiteStorage() = default;

  private:
    std::unordered_map<std::string, std::function<std::shared_ptr<ISuite>()>> m_storage_map;
  };

  class StoppingCriterionStorage {
  public:
    [[nodiscard]] auto has(const std::string &name) const -> bool {
      return m_storage_stopping.find(name) != m_storage_stopping.end();
    }
    /**
     *  throws std::out_of_range If the benchmark is not in the storage, use has_benchmark to check beforehand.
     */
    [[nodiscard]] auto at(const std::string &name) const
        -> std::function<std::unique_ptr<StoppingCriterion>(std::shared_ptr<Stats::StatsEngine>)> {
      return m_storage_stopping.at(name);
    };
    StoppingCriterionStorage() = default;

  private:
    std::unordered_map<std::string,
                       std::function<std::unique_ptr<StoppingCriterion>(std::shared_ptr<Stats::StatsEngine>)>>
        m_storage_stopping;
  };
} // namespace Baseliner
#endif // BASELINER_GENERAL_STORAGE_HPP