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
  void StatsEngine::build_execution_plan() { // NOLINT
    m_execution_plan.clear();

    std::unordered_map<std::type_index, IStatBase *> producers;

    for (const auto &stat : m_stats) {
      producers[stat->output()] = stat.get();
    }
    std::unordered_map<IStatBase *, int> dependency_count;

    // Queue of Stats that doesn't have computed depedencies
    std::vector<IStatBase *> ready_queue;
    for (const auto &stat : m_stats) {
      int count = 0;
      for (const auto &dep_type : stat->dependencies()) {
        // Check if this dependency is produced by another Stat
        if (producers.count(dep_type) > 0) {
          count++;
        }
        // If this depedency is an input value, we don't count it as depedency
      }
      dependency_count[stat.get()] = count;
      if (count == 0) {
        ready_queue.push_back(stat.get());
      }
    }

    // Process ready to go stats, and deliver other stats from their dependecy to this stat
    size_t processed_count = 0;
    while (!ready_queue.empty()) {
      // Pop a stat
      IStatBase *current = ready_queue.back();
      ready_queue.pop_back();

      // Add to plan
      m_execution_plan.push_back(current);
      processed_count++;

      auto current_output = current->output();

      for (const auto &neighbor : m_stats) {
        // Does 'neighbor' need what 'current' just produced?
        bool depends_on_current = false;
        for (auto dep : neighbor->dependencies()) {
          if (dep == current_output) {
            depends_on_current = true;
            break;
          }
        }

        if (depends_on_current) {
          dependency_count[neighbor.get()]--;
          if (dependency_count[neighbor.get()] == 0) {
            ready_queue.push_back(neighbor.get());
          }
        }
      }
    }

    if (processed_count != m_stats.size()) {
      throw std::runtime_error("Circular dependency detected in Stats Graph!");
    }

    m_is_built = true;
  };

  void StatsEngine::compute_stats() {
    build_execution_plan();
    for (IStatBase *stat : m_execution_plan) {
      stat->compute(m_registry);
    }
  };

} // namespace Baseliner::Stats