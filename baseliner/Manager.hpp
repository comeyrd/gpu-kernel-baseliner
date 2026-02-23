#ifndef BASELINER_MANAGER_HPP
#define BASELINER_MANAGER_HPP
#include <baseliner/Benchmark.hpp>
#include <baseliner/Case.hpp>
#include <functional>
#include <unordered_map>
namespace Baseliner {

  template <class BackendT>
  class Manager {
  private:
    std::unordered_map<std::string, std::function<std::shared_ptr<ICase<BackendT>>()>> _cases;
    std::unordered_map<std::string, std::function<std::shared_ptr<Benchmark<BackendT>>()>> _benchmark;

  public:
    static auto instance() -> Manager<BackendT> * {
      static Manager<BackendT> manager;
      return &manager;
    }

    auto get_case(std::string case_name) const -> std::shared_ptr<ICase<BackendT>> {
      if (_cases.find(case_name) != _cases.end()) {
        return _cases.at(case_name)();
      }
    }
    auto get_cases_names() -> std::vector<std::string> {
      std::vector<std::string> vecstr;
      for (auto [name, _] : _cases) {
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
    auto get_benchmarks_names() -> std::vector<std::string> {
      std::vector<std::string> vecstr;
      for (auto [name, _] : _benchmark) {
        vecstr.push_back(name);
      }
      return vecstr;
    }
    auto get_benchmark(std::string benchmark_name) const -> std::shared_ptr<Benchmark<BackendT>> {
      if (_benchmark.find(benchmark_name) != _benchmark.end()) {
        return _benchmark.at(benchmark_name)();
      }
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
#endif // BASELINER_MANAGER_HPP