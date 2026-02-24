#ifndef BASELINER_STOPPING_MANAGER_HPP
#define BASELINER_STOPPING_MANAGER_HPP
#include <baseliner/StoppingCriterion.hpp>
#include <baseliner/stats/StatsEngine.hpp>
#include <functional>
#include <unordered_map>
#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif
namespace Baseliner {

  class StoppingManager {
  private:
    std::unordered_map<std::string,
                       std::function<std::unique_ptr<StoppingCriterion>(std::shared_ptr<Stats::StatsEngine>)>>
        _stopping;

  public:
    static auto instance() -> StoppingManager * {
      static StoppingManager manager;
      return &manager;
    }

    auto get_stopping_recipe(const std::string &stopping_name)
        -> std::function<std::unique_ptr<StoppingCriterion>(std::shared_ptr<Stats::StatsEngine>)> {
      return inner_get_stopping(stopping_name);
    }

    [[nodiscard]] auto list_stopping() const -> std::vector<std::string> {
      std::vector<std::string> vecstr;
      for (const auto &[name, _] : _stopping) {
        vecstr.push_back(name);
      }
      return vecstr;
    }

    void register_stopping(
        std::string name,
        std::function<std::unique_ptr<StoppingCriterion>(std::shared_ptr<Stats::StatsEngine>)> stopping_recipe) {
      if (_stopping.find(name) == _stopping.end()) {
        _stopping[name] = stopping_recipe;
      }
    }

  private:
    auto inner_get_stopping(const std::string &stopping_name) const
        -> std::function<std::unique_ptr<StoppingCriterion>(std::shared_ptr<Stats::StatsEngine>)> {
      if (_stopping.find(stopping_name) != _stopping.end()) {
        return _stopping.at(stopping_name);
      }
      throw std::runtime_error("Stopping Criterion" + stopping_name + " not found");
    }
  };
  template <typename StoppingT>
  class StoppingRegistrar {
  public:
    explicit StoppingRegistrar(std::string name) {
      StoppingManager::instance()->register_stopping(
          name, [](std::shared_ptr<Stats::StatsEngine> engine) -> std::unique_ptr<StoppingCriterion> {
            return std::make_unique<StoppingT>(engine);
          });
    }
  };

#define BASELINER_REGISTER_STOPPING(Stopping)                                                                          \
  ATTRIBUTE_USED static Baseliner::StoppingRegistrar<Stopping> _registrar_##Stopping{#Stopping};
} // namespace Baseliner
#endif // BASELINER_STOPPING_MANAGER_HPP
