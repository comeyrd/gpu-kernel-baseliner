#ifndef BASELINER_BACKEND_MANAGER
#define BASELINER_BACKEND_MANAGER
#include "baseliner/Serializer.hpp"
#include "baseliner/managers/BackendSpecificStorage.hpp"
#include <baseliner/Benchmark.hpp>
#include <baseliner/Case.hpp>

namespace Baseliner {

  template <typename InnerTypeT>
  auto inject_preset(std::function<std::shared_ptr<InnerTypeT>()> &funct, const OptionsMap &preset)
      -> std::function<std::shared_ptr<InnerTypeT>()> {
    static_assert(std::is_base_of_v<IOption, InnerTypeT>,
                  "The type you want to inject presets on must inherit from IOption");
    auto output_function = [funct, preset]() -> std::shared_ptr<InnerTypeT> {
      auto ptr = funct();
      auto options = ptr->gather_options();
      if (Options::have_same_schema(options, preset)) {
        ptr->propagate_options(preset);
      } else {
        std::stringstream string_stream{};
        string_stream << "Error, presets should exactly match the object Option Schema \n";
        string_stream << "The given preset : \n";
        serialize(string_stream, preset);
        string_stream << "\n" << "The object preset \n";
        serialize(string_stream, options);
        std::runtime_error(string_stream.str());
      }
      return ptr;
    };
  }

  class IBackendStorage {
  public:
    virtual ~IBackendStorage() = default;

    [[nodiscard]] virtual auto get_benchmark_with_case(const std::string &benchmark_name,
                                                       const OptionsMap &benchmark_preset, const std::string &case_name,
                                                       const OptionsMap &case_preset,
                                                       const std::vector<std::string> &stats_names)
        -> std::function<std::shared_ptr<IBenchmark>()> = 0;

  protected:
    IBackendStorage() = default;
  };

  template <typename BackendT>
  class BackendStorage : public IBackendStorage {
  public:
    static auto instance() -> BackendStorage<BackendT> * {
      static BackendStorage<BackendT> manager;
      return &manager;
    }
    [[nodiscard]] auto get_benchmark_with_case(const std::string &benchmark_name, const OptionsMap &benchmark_preset,
                                               const std::string &case_name, const OptionsMap &case_preset,
                                               const std::vector<std::string> &stats_names)
        -> std::function<std::shared_ptr<IBenchmark>()> override {
      if (!m_benchmark_storage.has(benchmark_name)) {
        throw std::runtime_error("Benchmark " + benchmark_name + " does not exist in Backend" + m_name);
      }
      if (!m_cases_storage.has(case_name)) {
        throw std::runtime_error("Case " + case_name + " does not exist in Backend" + m_name);
      }
      auto benchmark_recipe = inject_preset(m_benchmark_storage.at(benchmark_name), benchmark_preset);
      auto case_recipe = inject_preset(m_cases_storage.at(case_name), case_preset);
      std::vector<std::function<void(std::shared_ptr<Stats::StatsEngine>)>> stats_recipes;
      for (auto stat : stats_names) {
        if (!m_backend_stats_storage.has(stat)) {
          throw std::runtime_error("Stat " + stat + " does not exist either in General Manager nor in Backend " +
                                   m_name);
        }
        stats_recipes.push_back(m_backend_stats_storage.at(stat));
      }
      auto func = [benchmark_recipe, case_recipe, stats_recipes]() -> std::shared_ptr<IBenchmark> {
        auto bench = benchmark_recipe();
        bench->set_case(case_recipe());
        bench->add_stats(stats_recipes);
        return bench;
      };
      return func;
    };

  protected:
    void set_name(const std::string &name) {
      m_name = name;
    }

  private:
    std::string m_name;
    CaseStorage<BackendT> m_cases_storage;
    BenchmarkStorage<BackendT> m_benchmark_storage;
    BackendStatsStorage<BackendT> m_backend_stats_storage;
    BackendStorage<BackendT>() = default;
  };

} // namespace Baseliner
#endif // BASELINER_BACKEND_MANAGER