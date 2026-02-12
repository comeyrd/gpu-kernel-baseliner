#include <baseliner/stats/IStats.hpp>
#include <baseliner/stats/StatsEngine.hpp>
#include <cstddef>
#include <stdexcept>
#include <typeindex>
#include <unordered_map>
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

  void StatsEngine::build_execution_plan() {
    m_execution_plan.clear();
    m_metric_to_stats.clear();
    m_unlinked_stats.clear();

    // 1. Identify all Metric Tags
    std::unordered_set<std::type_index> metric_tags;
    std::unordered_map<std::type_index, IMetricBase *> tag_to_metric;
    for (const auto &metric : m_metrics) {
      metric_tags.insert(metric->output());
      tag_to_metric[metric->output()] = metric.get();
    }

    // 2. Map Producers (Who provides what Tag)
    std::unordered_map<std::type_index, IStatBase *> producers;
    for (const auto &stat : m_stats) {
      producers[stat->output()] = stat.get();
    }

    // 3. Trace Links from Stats to Metrics
    // We check every Stat: "Do you (or your children) eventually consume a Metric Tag?"
    std::unordered_set<IStatBase *> linked_stats;

    // Recursive helper: returns true if this tag or its deps reach a Metric
    std::unordered_map<IStatBase *, bool> memo;
    std::function<bool(IStatBase *)> check_link = [&](IStatBase *stat) -> bool {
      if (memo.count(stat))
        return memo[stat];

      bool reaches_metric = false;
      for (const auto &dep_type : stat->dependencies()) {
        // Does this Stat directly consume a Metric?
        if (metric_tags.count(dep_type)) {
          m_metric_to_stats[tag_to_metric[dep_type]].push_back(stat);
          reaches_metric = true;
        }
        // Or does it consume another Stat that eventually leads to a Metric?
        else if (producers.count(dep_type)) {
          if (check_link(producers[dep_type])) {
            // If the child reaches a metric, then this parent does too.
            // We associate this stat with the same metrics its children have.
            for (const auto &metric_ptr : m_metrics) {
              auto &v = m_metric_to_stats[metric_ptr.get()];
              auto &child_v = m_metric_to_stats[metric_ptr.get()];
              // Note: To keep it clean, we verify the child is in that metric's list
              // This ensures transitive mapping: Median -> Sorted -> Metric
            }
            reaches_metric = true;
          }
        }
      }

      if (reaches_metric)
        linked_stats.insert(stat);
      return memo[stat] = reaches_metric;
    };

    // Correcting the transitive mapping logic to satisfy "Median depends on Metric"
    for (const auto &stat : m_stats) {
      for (const auto &dep_type : stat->dependencies()) {
        if (metric_tags.count(dep_type)) {
          // Direct link
          IStatBase *current = stat.get();
          IMetricBase *root_metric = tag_to_metric[dep_type];

          // We now need to bubble this "Metric-ness" up to anyone who consumes 'current'
          // But since we want "Median -> Sorted -> Metric", we use a simple discovery
          linked_stats.insert(current);
          m_metric_to_stats[root_metric].push_back(current);
        }
      }
    }

    // Propagate the links: If Stat A consumes Stat B, and B is linked to Metric X, then A is linked to X
    bool changed = true;
    while (changed) {
      changed = false;
      for (const auto &stat : m_stats) {
        for (const auto &dep_type : stat->dependencies()) {
          if (producers.count(dep_type)) {
            IStatBase *dependency = producers[dep_type];
            for (const auto &metric : m_metrics) {
              auto &vec = m_metric_to_stats[metric.get()];
              // If my dependency is linked to a metric, and I'm not yet...
              if (std::find(vec.begin(), vec.end(), dependency) != vec.end()) {
                if (std::find(vec.begin(), vec.end(), stat.get()) == vec.end()) {
                  vec.push_back(stat.get());
                  linked_stats.insert(stat.get());
                  changed = true;
                }
              }
            }
          }
        }
      }
    }

    // 4. Identify Unlinked Stats
    for (const auto &stat : m_stats) {
      if (linked_stats.find(stat.get()) == linked_stats.end()) {
        m_unlinked_stats.push_back(stat.get());
      }
    }

    // 5. Kahn's Algorithm (Standard Execution Order)
    std::unordered_map<std::type_index, std::vector<IStatBase *>> consumers;
    std::unordered_map<IStatBase *, int> dependency_count;
    std::vector<IStatBase *> ready_queue;

    for (const auto &stat : m_stats) {
      int count = 0;
      for (const auto &dep_type : stat->dependencies()) {
        if (producers.count(dep_type)) {
          count++;
          consumers[dep_type].push_back(stat.get());
        }
      }
      dependency_count[stat.get()] = count;
      if (count == 0)
        ready_queue.push_back(stat.get());
    }

    while (!ready_queue.empty()) {
      IStatBase *current = ready_queue.back();
      ready_queue.pop_back();
      m_execution_plan.push_back(current);
      if (consumers.count(current->output())) {
        for (IStatBase *neighbor : consumers[current->output()]) {
          if (--dependency_count[neighbor] == 0)
            ready_queue.push_back(neighbor);
        }
      }
    }

    m_is_built = true;
  }
  void StatsEngine::compute_stats() {
    build_execution_plan();
    for (IStatBase *stat : m_execution_plan) {
      stat->compute(m_registry);
    }
  };

} // namespace Baseliner::Stats