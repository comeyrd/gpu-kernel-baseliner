#ifndef BASELINER_BACKEND_MANAGER
#define BASELINER_BACKEND_MANAGER
#include "baseliner/Error.hpp"
#include <baseliner/Benchmark.hpp>
#include <baseliner/Case.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/managers/BackendSpecificStorage.hpp>
#include <baseliner/managers/Factories.hpp>
#include <baseliner/managers/PresetInjection.hpp>
namespace Baseliner {

  class IBackendStorage {
  public:
    virtual ~IBackendStorage() = default;

    [[nodiscard]] virtual auto get_benchmark_with_case(const std::string &benchmark_name,
                                                       const std::string &case_name) const -> IBenchmarkFactory = 0;

    void set_name(const std::string &name) {
      m_name = name;
    }
    [[nodiscard]] auto get_name() const -> std::string {
      return m_name;
    }
    [[nodiscard]] virtual auto list_device_stats() const -> std::vector<std::string> = 0;
    [[nodiscard]] virtual auto list_device_cases() const -> std::vector<std::string> = 0;
    [[nodiscard]] virtual auto list_device_benchmarks() const -> std::vector<std::string> = 0;
    [[nodiscard]] virtual auto list_components() -> ComponentList = 0;
    [[nodiscard]] virtual auto has_case(const std::string &name) const -> bool = 0;
    [[nodiscard]] virtual auto has_benchmark(const std::string &name) const -> bool = 0;
    [[nodiscard]] virtual auto has_stat(const std::string &name) const -> bool = 0;
    IBackendStorage() = default;
    virtual void apply_backend_preset(const OptionsMap &option) = 0;
    [[nodiscard]] virtual auto get_backend_setup(const OptionsMap &option) -> BackendSetup = 0;

  private:
    std::string m_name;
  };

  template <typename BackendT>
  class BackendStorage : public IBackendStorage {
  public:
    static auto instance() -> BackendStorage<BackendT> * {
      static BackendStorage<BackendT> manager;
      return &manager;
    }
    [[nodiscard]] auto get_benchmark_with_case(const std::string &benchmark_name, const std::string &case_name) const
        -> IBenchmarkFactory override {
      if (!m_benchmark_storage.has(benchmark_name)) {
        throw Errors::case_benchmark_not_found_in_backend(component_to_string(ComponentType::BENCHMARK), benchmark_name,
                                                          this->get_name());
      }
      if (!m_cases_storage.has(case_name)) {
        throw Errors::case_benchmark_not_found_in_backend(component_to_string(ComponentType::CASE), case_name,
                                                          this->get_name());
      }
      auto benchmark_recipe = m_benchmark_storage.at(benchmark_name);
      auto case_recipe = m_cases_storage.at(case_name);
      auto func = [benchmark_recipe, case_recipe]() -> std::shared_ptr<IBenchmark> {
        std::shared_ptr<Benchmark<BackendT>> bench = benchmark_recipe();
        bench->set_case(case_recipe());
        return bench;
      };
      return func;
    };
    void register_case(const std::string &name, const CaseFactory<BackendT> &case_factory) {
      m_cases_storage.insert(name, case_factory, get_name());
    }
    void register_benchmark(const std::string &name, const BenchmarkFactory<BackendT> &bench_factory) {
      m_benchmark_storage.insert(name, bench_factory, get_name());
    }
    void register_backend_stats(const std::string &name, const StatsFactory &stats_factory) {
      m_backend_stats_storage.insert(name, stats_factory, get_name());
    }
    [[nodiscard]] auto list_device_stats() const -> std::vector<std::string> override {
      return m_backend_stats_storage.list();
    };
    [[nodiscard]] auto list_device_cases() const -> std::vector<std::string> override {
      return m_cases_storage.list();
    };
    [[nodiscard]] auto list_device_benchmarks() const -> std::vector<std::string> override {
      return m_benchmark_storage.list();
    };
    [[nodiscard]] auto list_components() -> ComponentList override {
      ComponentType component_case = ComponentType::CASE;
      std::vector<std::pair<std::string, ComponentType>> result;
      result.reserve(m_cases_storage.size());
      for (const auto &str : list_device_cases()) {
        result.emplace_back(str, component_case);
      }
      ComponentType component_benchmark = ComponentType::BENCHMARK;
      result.reserve(result.size() + m_benchmark_storage.size());
      for (const auto &str : list_device_benchmarks()) {
        result.emplace_back(str, component_benchmark);
      }
      return result;
    };

    [[nodiscard]] auto has_case(const std::string &name) const -> bool override {
      return m_cases_storage.has(name);
    };
    [[nodiscard]] auto has_benchmark(const std::string &name) const -> bool override {
      return m_benchmark_storage.has(name);
    };
    [[nodiscard]] auto has_stat(const std::string &name) const -> bool override {
      return m_backend_stats_storage.has(name);
    };
    [[nodiscard]] auto get_backend_setup(const OptionsMap &option) -> BackendSetup override {
      BackendT *ptr = BackendT::instance();
      return [ptr, option]() {
        auto opt = ptr->gather_options();
        if (Options::is_subset(opt, option)) {
          ptr->propagate_options(option);
        } else {
          throw Errors::preset_not_subset_of(option, opt);
        }
      };
    }

  private:
    CaseStorage<BackendT> m_cases_storage;
    BenchmarkStorage<BackendT> m_benchmark_storage;
    BackendStatsStorage<BackendT> m_backend_stats_storage;
    BackendStorage<BackendT>() = default;
  };

} // namespace Baseliner
#endif // BASELINER_BACKEND_MANAGER