#ifndef BASELINER_MANAGER_HPP
#define BASELINER_MANAGER_HPP
#include <baseliner/Benchmark.hpp>
#include <baseliner/Case.hpp>
#include <functional>
#include <unordered_map>
namespace Baseliner {

  class IBackendManager {
  public:
    virtual ~IBackendManager() = default;

    virtual auto get_benchmark(const std::string &name) -> std::shared_ptr<IBenchmark> = 0;

    virtual void assign_case_to_benchmark(std::shared_ptr<IBenchmark> bench, const std::string &case_name) = 0;
    virtual auto get_benchmark_with_case(const std::string &benchmark_name, const std::string &case_name)
        -> std::shared_ptr<IBenchmark> = 0;
    [[nodiscard]] virtual auto list_benchmarks() const -> std::vector<std::string> = 0;
    [[nodiscard]] virtual auto list_cases() const -> std::vector<std::string> = 0;
  };

  template <class BackendT>
  class Manager : public IBackendManager {
  private:
    std::unordered_map<std::string, std::function<std::shared_ptr<ICase<BackendT>>()>> _cases;
    std::unordered_map<std::string, std::function<std::shared_ptr<Benchmark<BackendT>>()>> _benchmark;

  public:
    static auto instance() -> Manager<BackendT> * {
      static Manager<BackendT> manager;
      return &manager;
    }
    auto get_benchmark(const std::string &name) -> std::shared_ptr<IBenchmark> override {
      return inner_get_benchmark(name);
    };
    auto get_benchmark_with_case(const std::string &benchmark_name, const std::string &case_name)
        -> std::shared_ptr<IBenchmark> override {
      auto bench = inner_get_benchmark(benchmark_name);
      bench->set_case(inner_get_case(case_name));
      return bench;
    }
    void assign_case_to_benchmark(std::shared_ptr<IBenchmark> bench, const std::string &case_name) override {
      auto specific_bench = std::dynamic_pointer_cast<Benchmark<BackendT>>(bench);
      if (!specific_bench) {
        throw std::runtime_error("Invalid benchmark type passed to backend manager.");
      }
      specific_bench->set_case(inner_get_case(case_name));
    }

    [[nodiscard]] auto list_cases() const -> std::vector<std::string> override {
      std::vector<std::string> vecstr;
      for (auto [name, _] : _cases) {
        vecstr.push_back(name);
      }
      return vecstr;
    }
    [[nodiscard]] auto list_benchmarks() const -> std::vector<std::string> override {
      std::vector<std::string> vecstr;
      for (auto [name, _] : _benchmark) {
        vecstr.push_back(name);
      }
      return vecstr;
    }

    void register_case(std::string name, std::function<std::shared_ptr<ICase<BackendT>>()> case_recipe) {
      if (_cases.find(name) == _cases.end()) {
        _cases[name] = case_recipe;
      }
    }
    void register_benchmark(std::string name, std::function<std::shared_ptr<Benchmark<BackendT>>()> benchmark_recipe) {
      if (_benchmark.find(name) == _benchmark.end()) {
        _benchmark[name] = benchmark_recipe;
      }
    }

  private:
    auto inner_get_benchmark(std::string benchmark_name) const -> std::shared_ptr<Benchmark<BackendT>> {
      if (_benchmark.find(benchmark_name) != _benchmark.end()) {
        return _benchmark.at(benchmark_name)();
      }
      throw std::runtime_error("Benchmark" + benchmark_name + " not found");
    }
    auto inner_get_case(std::string case_name) const -> std::shared_ptr<ICase<BackendT>> {
      if (_cases.find(case_name) != _cases.end()) {
        return _cases.at(case_name)();
      }
      throw std::runtime_error("Case" + case_name + " not found");
    }
  };
  class BackendRegistry {
  public:
    static void register_backend(const std::string &name, IBackendManager *manager) {
      get_backends()[name] = manager;
    }

    static IBackendManager *get(const std::string &name) {
      auto &backends = get_backends();
      if (backends.count(name) > 0) {
        return backends[name];
      }
      throw std::runtime_error("Backend not available/compiled: " + name);
    }
    static auto list_backends() -> std::vector<std::string> {
      std::vector<std::string> vecstr;
      for (auto [name, _] : get_backends()) {
        vecstr.push_back(name);
      }
      return vecstr;
    }

  private:
    static auto get_backends() -> std::unordered_map<std::string, IBackendManager *> & {
      static std::unordered_map<std::string, IBackendManager *> backends;
      return backends;
    }
  };

  template <class BackendT>
  class BackendRegistrar {
  public:
    explicit BackendRegistrar(std::string name) {
      auto instance = Manager<BackendT>::instance();
      BackendRegistry::register_backend(name, instance);
    }
  };

  template <class KernelT>
  class KernelRegistrar {
  public:
    explicit KernelRegistrar(std::string name) {
      Manager<typename KernelT::backend>::instance()->register_case(
          name, []() -> std::shared_ptr<ICase<typename KernelT::backend>> {
            return std::make_shared<KernelCase<KernelT>>();
          });
    }
  };

  template <class CaseT>
  class CaseRegistrar {
  public:
    explicit CaseRegistrar(std::string name) {
      Manager<typename CaseT::backend>::instance()->register_case(
          name, []() -> std::shared_ptr<ICase<typename CaseT::backend>> { return std::make_shared<CaseT>(); });
    }
  };

  template <class BenchmarkT>
  class BenchmarkRegistrar {
  public:
    explicit BenchmarkRegistrar(std::string name) {
      Manager<typename BenchmarkT::backend>::instance()->register_benchmark(
          name,
          []() -> std::shared_ptr<Benchmark<typename BenchmarkT::backend>> { return std::make_shared<BenchmarkT>(); });
    }
  };

} // namespace Baseliner
#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif
#define BASELINER_REGISTER_KERNEL(KernelClass)                                                                         \
  ATTRIBUTE_USED static Baseliner::KernelRegistrar<KernelClass> _registrar_##KernelClass{#KernelClass};
#define BASELINER_REGISTER_CASE(CaseClass)                                                                             \
  ATTRIBUTE_USED static Baseliner::CaseRegistrar<CaseClass> _registrar_##CaseClass{#CaseClass};
#define BASELINER_REGISTER_BENCHMARK(BenchmarkClass)                                                                   \
  ATTRIBUTE_USED static Baseliner::BenchmarkRegistrar<BenchmarkClass> _registrar_##BenchmarkClass{#BenchmarkClass};

#define BASELINER_REGISTER_BACKEND(BackendClass, name)                                                                 \
  ATTRIBUTE_USED static Baseliner::BackendRegistrar<BackendClass> _registrar_##BackendClass{name};
#endif // BASELINER_MANAGER_HPP