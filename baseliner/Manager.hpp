#ifndef BASELINER_MANAGER_HPP
#define BASELINER_MANAGER_HPP
#include <baseliner/Benchmark.hpp>
#include <baseliner/Case.hpp>
namespace Baseliner {

  template <class BackendT>
  class Manager {
  private:
    std::unordered_map<std::string, std::shared_ptr<Benchmark<BackendT>>> _benchmarks;
    std::unordered_map<std::string, std::shared_ptr<ICase<BackendT>>> _cases;

  public:
    static auto instance() -> Manager<BackendT> * {
      static Manager<BackendT> manager;
      return &manager;
    }

    auto get_benchmarks() -> std::unordered_map<std::string, std::shared_ptr<Benchmark<BackendT>>> {
      return _benchmarks;
    }
    auto get_cases() -> std::unordered_map<std::string, std::shared_ptr<ICase<BackendT>>> {
      return _cases;
    }
    void register_benchmark(std::shared_ptr<Benchmark<BackendT>> bench_impl) {
      if (_benchmarks.find(bench_impl->name()) == _benchmarks.end()) {
        _benchmarks[bench_impl->name()] = bench_impl;
      }
    };
    void register_case(std::shared_ptr<ICase<BackendT>> case_impl) {
      if (_cases.find(case_impl->name()) == _cases.end()) {
        _cases[case_impl->name()] = case_impl;
      }
    }
    template <class KernelT>
    void register_kernel(std::shared_ptr<KernelCase<KernelT>> case_impl) {
      if (_cases.find(case_impl->name()) == _cases.end()) {
        _cases[case_impl->name()] = case_impl;
      }
    }
  };

  template <class KernelT>
  class KernelRegistrar {
  public:
    explicit KernelRegistrar() {
      Manager<typename KernelT::backend>::instance()->register_kernel(std::make_shared<KernelCase<KernelT>>());
    }
  };

  template <class BenchmarkT>
  class BenchmarkRegistrar {
  public:
    explicit BenchmarkRegistrar() {
      Manager<typename BenchmarkT::backend>::instance()->register_benchmark(std::make_shared<BenchmarkT>());
    }
  };

  template <class CaseT>
  class CaseRegistrar {
  public:
    explicit CaseRegistrar() {
      Manager<typename CaseT::backend>::instance()->register_case(std::make_shared<CaseT>());
    }
  };

} // namespace Baseliner
#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif
#define RegisterKernel(KernelClass)                                                                                    \
  ATTRIBUTE_USED static Baseliner::KernelRegistrar<KernelClass> _registrar_##KernelClass{};
#define RegisterCase(CaseClass) ATTRIBUTE_USED static Baseliner::CaseRegistrar<CaseClass> _registrar_##CaseClass{};
#define RegisterBenchmark(BenchmarkClass)                                                                              \
  ATTRIBUTE_USED static Baseliner::BenchmarkRegistrar<BenchmarkClass> _registrar_##BenchmarkClass{};
#endif // BASELINER_MANAGER_HPP