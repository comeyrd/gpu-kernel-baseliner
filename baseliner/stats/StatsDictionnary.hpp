#ifndef BASELINER_STATS_ABSTRACTION_HPP
#define BASELINER_STATS_ABSTRACTION_HPP
#include <baseliner/stats/StatsEngine.hpp>
#include <functional>
#include <string>
#include <unordered_map>
namespace Baseliner::Stats {

  class StatsDictionnary {
  private:
    std::unordered_map<std::string, std::function<void(std::shared_ptr<StatsEngine>)>> registerables;

  public:
    static auto instance() -> StatsDictionnary * {
      static StatsDictionnary dict;
      return &dict;
    }
    void register_stat(std::string name, std::function<void(std::shared_ptr<StatsEngine>)> stat_recipe) {
      if (registerables.find(name) == registerables.end()) {
        registerables[name] = stat_recipe;
      }
    }
    void add_stat_to_engine(std::string name, std::shared_ptr<StatsEngine> engine) {
      if (registerables.find(name) != registerables.end()) {
        registerables[name](engine);
      }
    }
    auto list_stats() -> std::vector<std::string> {
      std::vector<std::string> vectsr;
      for (auto [name, _] : registerables) {
        vectsr.push_back(name);
      }
      return vectsr;
    }
  };

  template <class StatT>
  class StatRegistrar {
  public:
    explicit StatRegistrar(std::string name) {
      StatsDictionnary::instance()->register_stat(
          name, [](std::shared_ptr<StatsEngine> engine) -> void { engine->register_stat<StatT>(); });
    }
  };

} // namespace Baseliner::Stats
#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif
#define BASELINER_REGISTER_STAT(StatClass)                                                                             \
  ATTRIBUTE_USED static Baseliner::Stats::StatRegistrar<StatClass> _registrar_##StatClass{#StatClass};
#endif // BASELINER_STATS_ABSTRACTION_HPP