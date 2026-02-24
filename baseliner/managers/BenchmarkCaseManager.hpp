#ifndef BASELINER_BENHCMARK_CASE_MANAGER_HPP
#define BASELINER_BENHCMARK_CASE_MANAGER_HPP
#include <baseliner/Benchmark.hpp>
#include <baseliner/Case.hpp>
#include <functional>
#include <unordered_map>
#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif
namespace Baseliner {
  class IBenchmarkCaseManager {
  public:
    virtual ~IBenchmarkCaseManager() = default;

    virtual auto get_benchmark_with_case(const std::string &benchmark_name, const std::string &case_name)
        -> std::function<std::shared_ptr<IBenchmark>()> = 0;
    [[nodiscard]] virtual auto list_benchmarks() const -> std::vector<std::string> = 0;
    [[nodiscard]] virtual auto list_cases() const -> std::vector<std::string> = 0;
  };

  template <class BackendT>
  class BenchmarkCaseManager : public IBenchmarkCaseManager {
  private:
    std::unordered_map<std::string, std::function<std::shared_ptr<ICase<BackendT>>()>> _cases;
    std::unordered_map<std::string, std::function<std::shared_ptr<Benchmark<BackendT>>()>> _benchmark;

  public:
    static auto instance() -> BenchmarkCaseManager<BackendT> * {
      static BenchmarkCaseManager<BackendT> manager;
      return &manager;
    }
    auto get_benchmark_with_case(const std::string &benchmark_name, const std::string &case_name)
        -> std::function<std::shared_ptr<IBenchmark>()> override {
      auto func = [this, benchmark_name, case_name]() -> std::shared_ptr<IBenchmark> {
        auto bench = inner_get_benchmark(benchmark_name)();
        bench->set_case(inner_get_case(case_name)());
        return bench;
      };
      return func;
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
    auto inner_get_benchmark(std::string benchmark_name) const
        -> std::function<std::shared_ptr<Benchmark<BackendT>>()> {
      if (_benchmark.find(benchmark_name) != _benchmark.end()) {
        return _benchmark.at(benchmark_name);
      }
      throw std::runtime_error("Benchmark" + benchmark_name + " not found");
    }
    auto inner_get_case(std::string case_name) const -> std::function<std::shared_ptr<ICase<BackendT>>()> {
      if (_cases.find(case_name) != _cases.end()) {
        return _cases.at(case_name);
      }
      throw std::runtime_error("Case" + case_name + " not found");
    }
  };

  template <class CaseT>
  class CaseRegistrar {
  public:
    explicit CaseRegistrar(std::string name) {
      BenchmarkCaseManager<typename CaseT::backend>::instance()->register_case(
          name, []() -> std::shared_ptr<ICase<typename CaseT::backend>> { return std::make_shared<CaseT>(); });
    }
    explicit CaseRegistrar(std::string name, std::function<std::shared_ptr<ICase<typename CaseT::backend>>()> builder) {
      BenchmarkCaseManager<CaseT>::instance()->register_case(name, builder);
    }
  };

  template <class BenchmarkT>
  class BenchmarkRegistrar {
  public:
    explicit BenchmarkRegistrar(std::string name) {
      BenchmarkCaseManager<typename BenchmarkT::backend>::instance()->register_benchmark(
          name,
          []() -> std::shared_ptr<Benchmark<typename BenchmarkT::backend>> { return std::make_shared<BenchmarkT>(); });
    }
    explicit BenchmarkRegistrar(std::string name, std::function<std::shared_ptr<Benchmark<BenchmarkT>>()> builder) {
      BenchmarkCaseManager<BenchmarkT>::instance()->register_benchmark(name, builder);
    }
  };
  template <class KernelT>
  class KernelRegistrar {
  public:
    explicit KernelRegistrar(std::string name) {
      BenchmarkCaseManager<typename KernelT::backend>::instance()->register_case(
          name, []() -> std::shared_ptr<ICase<typename KernelT::backend>> {
            return std::make_shared<KernelCase<KernelT>>();
          });
    }
  };
#define BASELINER_REGISTER_KERNEL(KernelClass)                                                                         \
  ATTRIBUTE_USED static Baseliner::KernelRegistrar<KernelClass> _registrar_##KernelClass{#KernelClass};
#define BASELINER_REGISTER_CASE(CaseClass)                                                                             \
  ATTRIBUTE_USED static Baseliner::CaseRegistrar<CaseClass> _registrar_##CaseClass{#CaseClass};
#define BASELINER_REGISTER_CASE_LAMBDA(Backend, Lambda, name)                                                          \
  ATTRIBUTE_USED static Baseliner::CaseRegistrar<Backend> _registrar_##CaseClass{name, Lambda};
#define BASELINER_REGISTER_BENCHMARK(BenchmarkClass)                                                                   \
  ATTRIBUTE_USED static Baseliner::BenchmarkRegistrar<BenchmarkClass> _registrar_##BenchmarkClass{#BenchmarkClass};
#define BASELINER_REGISTER_BENCHMARK_LAMBDA(Backend, Lambda, name)                                                     \
  ATTRIBUTE_USED static Baseliner::BenchmarkRegistrar<Backend> _registrar_##lambda##__LINE__{name, Lambda};
} // namespace Baseliner
#endif // BASELINER_BENHCMARK_CASE_MANAGER_HPP
