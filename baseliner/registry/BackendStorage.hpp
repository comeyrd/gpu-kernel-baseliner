#ifndef BASELINER_REGISTRY_BACKENDSTORAGE_HPP
#define BASELINER_REGISTRY_BACKENDSTORAGE_HPP
#include <baseliner/core/Benchmark.hpp>
#include <baseliner/core/Error.hpp>
#include <baseliner/core/Options.hpp>
#include <baseliner/core/Workload.hpp>
#include <baseliner/registry/BackendSpecificStorage.hpp>
#include <baseliner/registry/Factories.hpp>
#include <baseliner/registry/PresetInjection.hpp>
namespace Baseliner {

  class IBackendStorage {
  public:
    virtual ~IBackendStorage() = default;

    [[nodiscard]] virtual auto get_benchmark_with_workload(const std::string &benchmark_name,
                                                           const std::string &workload_name) const
        -> IBenchmarkFactory = 0;

    void set_name(const std::string &name) {
      m_name = name;
    }
    [[nodiscard]] auto get_name() const -> std::string {
      return m_name;
    }
    [[nodiscard]] virtual auto list_device_stats() const -> std::vector<std::string> = 0;
    [[nodiscard]] virtual auto list_device_stats_options() const -> std::unordered_map<std::string, OptionsMap> = 0;
    [[nodiscard]] virtual auto list_device_workloads() const -> std::vector<std::string> = 0;
    [[nodiscard]] virtual auto list_device_benchmarks() const -> std::vector<std::string> = 0;
    [[nodiscard]] virtual auto list_components() -> ComponentList = 0;
    [[nodiscard]] virtual auto has_workload(const std::string &name) const -> bool = 0;
    [[nodiscard]] virtual auto has_benchmark(const std::string &name) const -> bool = 0;
    [[nodiscard]] virtual auto has_stat(const std::string &name) const -> bool = 0;
    IBackendStorage() = default;

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
    [[nodiscard]] auto get_benchmark_with_workload(const std::string &benchmark_name,
                                                   const std::string &workload_name) const
        -> IBenchmarkFactory override {
      if (!m_benchmark_storage.has(benchmark_name)) {
        throw Errors::workload_benchmark_not_found_in_backend(component_to_string(ComponentType::BENCHMARK),
                                                              benchmark_name, this->get_name());
      }
      if (!m_workloads_storage.has(workload_name)) {
        throw Errors::workload_benchmark_not_found_in_backend(component_to_string(ComponentType::CASE), workload_name,
                                                              this->get_name());
      }
      auto benchmark_recipe = m_benchmark_storage.at(benchmark_name);
      auto workload_recipe = m_workloads_storage.at(workload_name);
      auto func = [benchmark_recipe, workload_recipe]() -> std::shared_ptr<IBenchmark> {
        std::shared_ptr<Benchmark<BackendT>> bench = benchmark_recipe();
        bench->set_workload(workload_recipe());
        return bench;
      };
      return func;
    };
    void register_workload(const std::string &name, const WorkloadFactory<BackendT> &workload_factory) {
      m_workloads_storage.insert(name, workload_factory, get_name());
    }
    void register_benchmark(const std::string &name, const BenchmarkFactory<BackendT> &bench_factory) {
      m_benchmark_storage.insert(name, bench_factory, get_name());
    }
    void register_backend_stats(const std::string &name, const StatsFactory &stats_factory, const OptionsMap &options) {
      m_backend_stats_storage.insert(name, stats_factory, get_name());
      if (!options.empty()) {
        m_backend_stats_storage.insert_options(name, options);
      }
    }
    [[nodiscard]] auto list_device_stats_options() const -> std::unordered_map<std::string, OptionsMap> override {
      return m_backend_stats_storage.list_w_options();
    };

    [[nodiscard]] auto list_device_stats() const -> std::vector<std::string> override {
      return m_backend_stats_storage.list();
    };
    [[nodiscard]] auto list_device_workloads() const -> std::vector<std::string> override {
      return m_workloads_storage.list();
    };
    [[nodiscard]] auto list_device_benchmarks() const -> std::vector<std::string> override {
      return m_benchmark_storage.list();
    };
    [[nodiscard]] auto list_components() -> ComponentList override {
      ComponentType component_workload = ComponentType::CASE;
      std::vector<std::pair<std::string, ComponentType>> result;
      result.reserve(m_workloads_storage.size());
      for (const auto &str : list_device_workloads()) {
        result.emplace_back(str, component_workload);
      }
      ComponentType component_benchmark = ComponentType::BENCHMARK;
      result.reserve(result.size() + m_benchmark_storage.size());
      for (const auto &str : list_device_benchmarks()) {
        result.emplace_back(str, component_benchmark);
      }
      return result;
    };

    [[nodiscard]] auto has_workload(const std::string &name) const -> bool override {
      return m_workloads_storage.has(name);
    };
    [[nodiscard]] auto has_benchmark(const std::string &name) const -> bool override {
      return m_benchmark_storage.has(name);
    };
    [[nodiscard]] auto has_stat(const std::string &name) const -> bool override {
      return m_backend_stats_storage.has(name);
    };

  private:
    WorkloadStorage<BackendT> m_workloads_storage;
    BenchmarkStorage<BackendT> m_benchmark_storage;
    BackendStatsStorage<BackendT> m_backend_stats_storage;
    BackendStorage<BackendT>() = default;
  };

} // namespace Baseliner
#endif // BASELINER_BACKEND_MANAGER