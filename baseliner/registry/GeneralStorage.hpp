#ifndef BASELINER_REGISTRY_GENERALSTORAGE_HPP
#define BASELINER_REGISTRY_GENERALSTORAGE_HPP
#include <baseliner/core/stats/StatsEngine.hpp>
#include <baseliner/registry/Factories.hpp>
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
    [[nodiscard]] auto at(const std::string &name) const -> StatsFactory {
      return m_stats_map.at(name);
    };

    void insert(const std::string &name, const StatsFactory &stat_func) {
      if (has(name)) {
        throw Errors::already_exist("Stat", name);
      }
      m_stats_map[name] = stat_func;
    }

    void insert_options(const std::string &name, const OptionsMap &options) {
      m_stats_options[name] = options;
    }
    GeneralStatsStorage() = default;

    [[nodiscard]] auto list() const -> std::vector<std::string> {
      std::vector<std::string> vecstr;
      vecstr.reserve(m_stats_map.size());
      for (const auto &[name, _] : m_stats_map) {
        vecstr.push_back(name);
      }
      return vecstr;
    }
    [[nodiscard]] auto list_w_options() const -> std::unordered_map<std::string, OptionsMap> {
      return m_stats_options;
    }

  private:
    std::unordered_map<std::string, StatsFactory> m_stats_map;
    std::unordered_map<std::string, OptionsMap> m_stats_options;
  };
  class StoppingCriterionStorage {
  public:
    [[nodiscard]] auto has(const std::string &name) const -> bool {
      return m_storage_stopping.find(name) != m_storage_stopping.end();
    }
    /**
     *  throws std::out_of_range If the benchmark is not in the storage, use has_benchmark to check beforehand.
     */
    [[nodiscard]] auto at(const std::string &name) const -> StoppingCriterionFactory {
      return m_storage_stopping.at(name);
    };
    void insert(const std::string &name, const StoppingCriterionFactory &stopping_func) {
      if (has(name)) {
        throw Errors::already_exist(component_to_string(STOPPING), name);
      }
      m_storage_stopping[name] = stopping_func;
    }
    StoppingCriterionStorage() = default;

  private:
    std::unordered_map<std::string, StoppingCriterionFactory> m_storage_stopping;
  };
} // namespace Baseliner
#endif // BASELINER_GENERAL_STORAGE_HPP