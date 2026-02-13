#ifndef BASELINER_STATS_ENGINE_HPP
#define BASELINER_STATS_ENGINE_HPP
#include <baseliner/Metric.hpp>
#include <baseliner/stats/IStats.hpp>
#include <baseliner/stats/StatsRegistry.hpp>
#include <cstddef>
#include <functional>
#include <memory>
#include <sstream>
#include <typeindex>
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
    auto operator()(const Baseliner::Stats::TypeIndexArgs &obj) const {
      return std::hash<std::type_index>{}(obj.type_ix);
    }
  };
} // namespace std
namespace Baseliner::Stats {
  class StatsEngine : public LazyOption {
  public:
    // Ownership is transferred to the Engine.
    // TODO link a Stat to a Metric
    // TODO check if everybody has it's depedencies fulfilled
    template <typename StatType, typename... Args>
    void register_stat(Args &&...args) {
      if (m_is_built) {
        throw std::runtime_error("Trying to register a metric after the Engine is built");
      }
      static_assert(std::is_base_of_v<IStatBase, StatType>, "Must derive from IStatBase");

      auto iterator = m_registered_types.find({std::type_index(typeid(StatType)), false});
      constexpr bool has_args = sizeof...(args) > 0;

      if (iterator != m_registered_types.end()) {
        if (!iterator->has_args && has_args) {
          inner_remove_stat<StatType>();
          inner_register_stat<StatType>(std::forward<Args>(args)...);
        }
      } else {
        inner_register_stat<StatType>(std::forward<Args>(args)...);
      }
    }
    template <typename MetricType, typename... Args>
    void register_metric(Args &&...args) {
      if (m_is_built) {
        throw std::runtime_error("Trying to register a metric after the Engine is built");
      }
      static_assert(std::is_base_of_v<IMetricBase, MetricType>, "Registered type must derive from IMetricBase");
      auto iterator = m_registered_types.find({std::type_index(typeid(MetricType)), false});
      const bool current_has_args = sizeof...(args) > 0;

      if (iterator != m_registered_types.end()) {
        if (!iterator->has_args) {
          if (current_has_args) {
            inner_remove_metric<MetricType>();
            inner_register_metric<MetricType>(std::forward<Args>(args)...);
          }
        }
      } else {
        inner_register_metric<MetricType>(std::forward<Args>(args)...);
      }
    };
    template <typename... InputTags>
    void register_stats_dependencies() {
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
      std::ostringstream oss;
      oss << "StatsEngine error in get_result(): " << typeid(StatType).name() << " Stat is not registered";
      throw std::runtime_error(oss.str());
    };
    void compute_stats();

    auto get_metrics() -> std::vector<Metric> {
      std::vector<Metric> metrics_vector{};
      for (auto &metric_ptr : m_metrics) {
        Metric metric;
        metric.m_name = metric_ptr->name();
        metric.m_unit = metric_ptr->unit();

        // Use the map we built in build_execution_plan
        if (m_metric_to_stats.count(metric_ptr.get()) > 0) {
          for (IStatBase *stat : m_metric_to_stats.at(metric_ptr.get())) {
            metric.m_v_stats.push_back({stat->name(), stat->get_value(m_registry)});
          }
        }
        metrics_vector.push_back(metric);
      }
      for (auto &stat : m_unlinked_stats) {
        Metric metric;
        metric.m_name = stat->name();
        metric.m_unit = std::string();
        metric.m_data = stat->get_value(m_registry);
        metrics_vector.push_back(metric);
      }
      return metrics_vector;
    }

    void register_options_dependencies() override {
      for (auto &stat : m_stats) {
        register_consumer(*stat);
      }
      for (auto &metric : m_metrics) {
        register_consumer(*metric);
      }
    };

    void reset() {
      ensure_build();
      m_registry = StatsRegistry();
      set_default();
    };

  private:
    template <typename... Ts>
    void expand_and_register(std::tuple<Ts...> * /*unused*/) {
      register_stats_dependencies<Ts...>();
    }
    template <typename StatType, typename... Args>
    void inner_register_stat(Args &&...args) {
      const bool current_has_args = sizeof...(args) > 0;
      auto new_stat = std::make_unique<StatType>(std::forward<Args>(args)...);
      m_stats.push_back(std::move(new_stat));
      m_registered_types.insert({std::type_index(typeid(StatType)), current_has_args});
      expand_and_register(static_cast<typename StatType::tuple *>(nullptr));
    };

    template <typename StatType>
    void inner_remove_stat() {
      auto test = [](std::unique_ptr<IStatBase> &basestat) { return (typeid(basestat->output()) == typeid(StatType)); };
      auto iterator = std::find_if(m_stats.begin(), m_stats.end(), test);
      if (iterator != m_stats.end()) {
        m_stats.erase(iterator);
      }
    }

    template <typename MetricType, typename... Args>
    void inner_register_metric(Args &&...args) {
      const bool current_has_args = sizeof...(args) > 0;
      auto new_metric = std::make_unique<MetricType>(std::forward<Args>(args)...);
      m_metrics.push_back(std::move(new_metric));
      m_registered_types.insert({std::type_index(typeid(MetricType)), current_has_args});
    };
    template <typename MetricType>
    void inner_remove_metric() {
      auto test = [](std::unique_ptr<IMetricBase> &basemetric) {
        return (typeid(basemetric->output()) == typeid(MetricType));
      };
      auto iterator = std::find_if(m_metrics.begin(), m_metrics.end(), test);
      if (iterator != m_metrics.end()) {
        m_metrics.erase(iterator);
      }
    }

    StatsRegistry m_registry;
    std::vector<std::unique_ptr<IMetricBase>> m_metrics;

    // The pool of all owned stat objects
    std::vector<std::unique_ptr<IStatBase>> m_stats;
    // The registered  stat types
    std::unordered_set<TypeIndexArgs> m_registered_types;
    std::vector<IStatBase *> m_unlinked_stats;
    // These point to the objects inside stats_.
    std::vector<IStatBase *> m_execution_plan;
    std::unordered_map<IMetricBase *, std::vector<IStatBase *>> m_metric_to_stats;
    bool m_is_built = false;
    void ensure_build();
    void build_execution_plan();
    void set_default();
  };
} // namespace Baseliner::Stats
#endif