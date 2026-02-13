#include <baseliner/stats/IStats.hpp>
#include <baseliner/stats/StatsEngine.hpp>
#include <cstddef>
#include <stdexcept>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
namespace Baseliner::Stats {

  void StatsEngine::ensure_build() {
    if (!m_is_built) {
      build_execution_plan();
      m_is_built = true;
    }
  };
  // TODO Simplify
  // TODO Build adjacency matrix ? with width and height all stats and all metrics ? easier to do lookups afterward.

  enum class NodeStatus : uint8_t {
    Unvisited,
    Visiting,
    Resolved
  };

  static void
  depth_first_search(std::type_index current,
                     std::unordered_map<std::type_index, std::unordered_set<std::type_index>> &producer_to_dep,
                     std::unordered_map<std::type_index, NodeStatus> &status) {
    NodeStatus &current_status = status[current];

    if (current_status == NodeStatus::Resolved) {
      return;
    }
    if (current_status == NodeStatus::Visiting) {
      throw std::runtime_error("Circular dependency detected in metrics graph!");
    }

    // 2. Mark as currently visiting (on the recursion stack)
    current_status = NodeStatus::Visiting;

    std::unordered_set<std::type_index> flattened_deps;

    // 3. Process dependencies
    if (producer_to_dep.count(current) > 0) {
      // We copy the immediate dependencies because we will be
      // overwriting producer_to_dep[current] later.
      auto immediate_deps = producer_to_dep.at(current);

      for (const auto &dep : immediate_deps) {
        // Recurse first to ensure the child is fully flattened
        depth_first_search(dep, producer_to_dep, status);

        // Add the child itself
        flattened_deps.insert(dep);

        // Add all of the child's now-flattened dependencies
        const auto &child_flattened = producer_to_dep.at(dep);
        flattened_deps.insert(child_flattened.begin(), child_flattened.end());
      }
    }

    // 4. Update the map with the flattened results and mark as Resolved
    producer_to_dep[current] = std::move(flattened_deps);
    current_status = NodeStatus::Resolved;
  }

  void StatsEngine::build_execution_plan() {
    m_execution_plan.clear();
    m_metric_to_stats.clear();
    m_unlinked_stats.clear();

    std::unordered_map<std::type_index, IMetricBase *> tag_to_metric;
    std::unordered_map<std::type_index, IStatBase *> tag_to_producers;

    std::unordered_map<std::type_index, std::unordered_set<std::type_index>> producer_to_depedency;
    std::unordered_map<std::type_index, std::unordered_set<std::type_index>> depedency_to_producer;

    // fill the metric maps
    for (const auto &metric : m_metrics) {
      tag_to_metric[metric->output()] = metric.get();
    }
    // create the map and the reverse map
    for (const auto &stat : m_stats) {
      // filling the producers Map
      tag_to_producers[stat->output()] = stat.get();
      // filling producer_to_depedency & depedency_to_producer_map
      auto depedencies = stat->dependencies();
      producer_to_depedency[stat->output()].insert(depedencies.begin(), depedencies.end());
    }

    // We use a separate list of keys to avoid iterator invalidation on the map itself
    std::vector<std::type_index> keys;
    keys.reserve(producer_to_depedency.size());
    for (auto const &[key, _] : producer_to_depedency) {
      keys.push_back(key);
    }
    std::unordered_map<std::type_index, NodeStatus> status;
    for (const auto &key : keys) {
      depth_first_search(key, producer_to_depedency, status);
    }
    // we got the flattened producer_to_depedency map

    // now we create the reverse map
    for (auto &[producer, depedencies] : producer_to_depedency) {
      for (const auto &depedency : depedencies) {
        depedency_to_producer[depedency].insert(producer);
      }
    }
    // adding stats with no depedency to unlinked stats
    for (const auto &[producer_tag, ptr] : tag_to_producers) {
      if (producer_to_depedency[producer_tag].empty()) {
        m_unlinked_stats.push_back(ptr);
      }
    }

    // First remove metrics and fill metric_to_stats
    for (const auto &[metric_tag, ptr] : tag_to_metric) {
      for (const auto &stat : depedency_to_producer[metric_tag]) {
        producer_to_depedency[stat].erase(metric_tag);
        m_metric_to_stats[tag_to_metric[metric_tag]].push_back(tag_to_producers[stat]);
      }
      depedency_to_producer.erase(metric_tag);
    }

    auto not_in_execution_plan = tag_to_producers;
    while (!not_in_execution_plan.empty()) {
      for (auto &[producer, ptr] : tag_to_producers) {
        if (producer_to_depedency[producer].empty()) {
          m_execution_plan.push_back(ptr);
          not_in_execution_plan.erase(producer);
          for (const auto &depedencies : depedency_to_producer[producer]) {
            producer_to_depedency[depedencies].erase(producer);
          }
          depedency_to_producer.erase(producer);
        }
      }
      tag_to_producers = not_in_execution_plan;
    }
    if (m_execution_plan.size() != m_stats.size()) {
      throw std::runtime_error("Circular dependency detected in Stats Graph!");
    }

    m_is_built = true;
  }
  void StatsEngine::compute_stats() {
    ensure_build();
    for (IStatBase *stat : m_execution_plan) {
      stat->compute(m_registry);
    }
  };

} // namespace Baseliner::Stats