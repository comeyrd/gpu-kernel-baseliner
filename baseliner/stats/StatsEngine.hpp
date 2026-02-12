#ifndef BASELINER_STATS_ENGINE_HPP
#define BASELINER_STATS_ENGINE_HPP
#include <baseliner/Metric.hpp>
#include <baseliner/stats/IStats.hpp>
#include <baseliner/stats/StatsRegistry.hpp>
#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>
namespace Baseliner::Stats {
  struct TypeIndexArgs {
    std::type_index type_ix;
    bool has_args;
    bool operator==(const TypeIndexArgs &other) const {
      return (type_ix == other.type_ix);
    };
  };
} // namespace Baseliner::Stats
namespace std {
  template <>
  struct hash<Baseliner::Stats::TypeIndexArgs> {
    size_t operator()(const Baseliner::Stats::TypeIndexArgs &obj) const {
      return std::hash<std::type_index>{}(obj.type_ix);
    }
  };
} // namespace std
namespace Baseliner::Stats {
  class StatsEngine {
  public:
    // Ownership is transferred to the Engine.
    // TODO Autoregistry if not found
    // TODO link a Stat to a Metric
    template <typename StatType, typename... Args>
    void register_stat(Args &&...args) {
      static_assert(std::is_base_of_v<IStatBase, StatType>, "Must derive from IStatBase");

      auto iterator = m_registered_types.find({std::type_index(typeid(StatType)), false});
      constexpr bool has_args = sizeof...(args) > 0;

      if (iterator != m_registered_types.end()) {
        if (!iterator->has_args && has_args) {
          inner_register_type<StatType>(std::forward<Args>(args)...);
        }
      } else {
        inner_register_type<StatType>(std::forward<Args>(args)...);
      }
    }
    template <typename MetricType, typename... Args>
    void register_metric(Args &&...args) {
      static_assert(std::is_base_of_v<IMetricBase, MetricType>, "Registered type must derive from IMetricBase");
      auto iterator = m_registered_types.find({std::type_index(typeid(MetricType)), false});
      bool current_has_args = sizeof...(args) > 0;

      if (iterator != m_registered_types.end()) {
        if (!iterator->has_args) {
          if (current_has_args) {
            inner_register_metric<MetricType>(std::forward<Args>(args)...);
          }
        }
      } else {
        inner_register_metric<MetricType>(std::forward<Args>(args)...);
      }
    };
    template <typename... InputTags>
    void register_depedencies() {
      (
          [&] {
            if constexpr (std::is_base_of_v<IStatBase, InputTags>) {
              register_stat<InputTags>();
            }
            if constexpr (std::is_base_of_v<IMetricBase, InputTags>) {
              register_metric<InputTags>();
            } else {
            }
          }(),
          ...);
    }

    // Builds the execution_plan
    //  Must be called once before compute().
    void build_execution_plan();

    // Sets the source inputs
    template <typename... StatType>
    void update_values(typename StatType::type... value) {
      (m_registry.set<StatType>(value), ...);
    };

    // Access a result.
    template <typename StatType>
    auto get_result() const -> const typename StatType::type & {
      auto iterator = m_registered_types.find({std::type_index(typeid(StatType)), false});
      if (iterator != m_registered_types.end()) {
        return m_registry.get<StatType>();
      }
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
    template <typename StatType, typename... Args>
    void inner_register_type(Args &&...args) {
      bool current_has_args = sizeof...(args) > 0;
      auto new_stat = std::make_unique<StatType>(std::forward<Args>(args)...);
      m_stats.push_back(std::move(new_stat));
      m_registered_types.insert({std::type_index(typeid(StatType)), current_has_args});
      register_depedencies<typename StatType::tuple>();
    };

    template <typename MetricType, typename... Args>
    void inner_register_metric(Args &&...args) {
      bool current_has_args = sizeof...(args) > 0;
      auto new_metric = std::make_unique<MetricType>(std::forward<Args>(args)...);
      m_metrics.push_back(std::move(new_metric));
      m_registered_types.insert({std::type_index(typeid(MetricType)), current_has_args});
    };

    StatsRegistry m_registry;
    std::vector<std::unique_ptr<IMetricBase>> m_metrics;

    // The pool of all owned stat objects
    std::vector<std::unique_ptr<IStatBase>> m_stats;
    // The registered  stat types
    std::unordered_set<TypeIndexArgs> m_registered_types;

    // These point to the objects inside stats_.
    std::vector<IStatBase *> m_execution_plan;

    bool m_is_built = false;
    void ensure_build();
  };
} // namespace Baseliner::Stats
#endif