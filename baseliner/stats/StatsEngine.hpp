#ifndef BASELINER_STATS_ENGINE_HPP
#define BASELINER_STATS_ENGINE_HPP
#include <baseliner/Metric.hpp>
#include <baseliner/stats/IStats.hpp>
#include <baseliner/stats/StatsRegistry.hpp>
#include <memory>
#include <unordered_set>
#include <vector>
namespace Baseliner::Stats {
  class StatsEngine {
  public:
    // Ownership is transferred to the Engine.
    // TODO Autoregistry if not found
    template <typename StatType, typename... Args>
    void register_stat(Args &&...args) {
      static_assert(std::is_base_of_v<IStatBase, StatType>, "Registered type must derive from IStatBase");
      auto new_stat = std::make_unique<StatType>(std::forward<Args>(args)...);
      m_stats.push_back(std::move(new_stat));
      m_registered_types.insert(std::type_index(typeid(StatType)));
    };

    // Builds the execution_plan
    //  Must be called once before compute().
    void build_execution_plan();

    // Sets the source inputs
    template <typename... Tag>
    void update_values(typename Tag::type... value) {
      (m_registry.set<Tag>(value), ...);
    };

    // Access a result.
    template <typename Tag>
    auto get_result() const -> const typename Tag::type & {
      return m_registry.get<Tag>();
    };
    void compute_stats();

    auto get_metrics() -> Metric {
      Metric metric;
      metric.m_name = "execution_times";
      metric.m_unit = "ms";
      metric.m_v_stats = {};
      for (auto &stat : m_stats) {
        metric.m_v_stats.push_back({stat->name(), // TODO get the value ?
                                    stat->get_value(m_registry)});
      }
      return metric;
    };

  private:
    StatsRegistry m_registry;
    // goes through the execution plan

    // The pool of all owned stat objects
    std::vector<std::unique_ptr<IStatBase>> m_stats;
    // The registered  stat types
    std::unordered_set<std::type_index> m_registered_types;

    // These point to the objects inside stats_.
    std::vector<IStatBase *> m_execution_plan;

    bool m_is_built = false;
    void ensure_build();
  };
} // namespace Baseliner::Stats
#endif