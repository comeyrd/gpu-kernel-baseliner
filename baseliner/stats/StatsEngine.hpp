#ifndef _BASELINER_STATS_ENGINE_HPP
#define _BASELINER_STATS_ENGINE_HPP
#include <baseliner/stats/IStats.hpp>
#include <baseliner/stats/StatsRegistry.hpp>
#include <memory>
#include <vector>
namespace Baseliner::Stats {
  class StatsEngine {
  public:
    // Ownership is transferred to the Engine.
    template <typename StatType, typename... Args>
    void register_stat(Args &&...args) {
      static_assert(std::is_base_of_v<IStatBase, StatType>, "Registered type must derive from IStatBase");
      auto new_stat = std::make_unique<StatType>(std::forward<Args>(args)...);
      m_stats.push_back(std::move(new_stat));
    };

    // Builds the execution_plan
    //  Must be called once before compute().
    void build_execution_plan();

    // Sets the source inputs
    template <typename Tag>
    void set_input(typename Tag::type value) {
      m_registry.set<Tag>(value);
    };

    // Goes through the execution_plan
    void compute_pass();

    // Access a result.
    template <typename Tag>
    auto get_result() const -> const typename Tag::type & {
      return m_registry.get<Tag>();
    };

  private:
    StatsRegistry m_registry;

    // The pool of all owned stat objects
    std::vector<std::unique_ptr<IStatBase>> m_stats;

    // These point to the objects inside stats_.
    std::vector<IStatBase *> m_execution_plan;

    bool m_is_built = false;
    void ensure_build();
  };
} // namespace Baseliner::Stats
#endif