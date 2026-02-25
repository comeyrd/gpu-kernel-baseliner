#ifndef BASELINER_REGISTRARS_HPP
#define BASELINER_REGISTRARS_HPP
#include <baseliner/managers/BackendStorage.hpp>
#include <baseliner/managers/Manager.hpp>
#include <memory>
namespace Baseliner {

  // Also registers every default thing

  template <class StatT>
  class GeneralStatRegistrar {
  public:
    explicit GeneralStatRegistrar(const std::string &name) {
      Manager::instance()->register_general_stat(
          name, [](std::shared_ptr<Stats::StatsEngine> engine) { engine->register_stat<StatT>(); });
    }
  };

  template <class SuiteT>
  class SuiteRegistrar {
  public:
    explicit SuiteRegistrar(const std::string &name) {
      Manager::instance()->register_suite(name, []() -> std::shared_ptr<ISuite> { return std::make_shared<SuiteT>(); });
    }
  };

  template <class StoppingT>
  class StoppingRegistrar {
  public:
    explicit StoppingRegistrar(const std::string &name) {
      Manager::instance()->register_stopping(
          name, []() -> std::unique_ptr<StoppingCriterion> { return std::make_unique<StoppingT>(); });
    }
  };

  template <class BenchmarkT>
  class BenchmarkRegistrar {
  public:
    explicit BenchmarkRegistrar(const std::string &name) {
      auto factory = []() -> std::shared_ptr<Benchmark<typename BenchmarkT::backend>> {
        return std::make_shared<BenchmarkT>();
      };
      BackendStorage<typename BenchmarkT::backend>::instance()->register_benchmark(name, factory);
      Manager::instance()->add_preset(name, DEFAULT_PRESET, {DEFAULT_DESCRIPTION, factory()->gather_options()},
                                      ComponentType::BENCHMARK);
    }
  };

  template <class BackendT>
  class BackendRegistrar {
  public:
    explicit BackendRegistrar(const std::string &name) {
      IBackendStorage *backend = BackendStorage<BackendT>::instance();
      backend->set_name(name);
      Manager::instance()->register_backend(name, backend);
      BenchmarkRegistrar<Benchmark<BackendT>> register_me("Benchmark");
    }
  };

  template <class CaseT>
  class CaseRegistrar {
  public:
    explicit CaseRegistrar(const std::string &name) {
      auto factory = []() -> std::shared_ptr<ICase<typename CaseT::backend>> { return std::make_shared<CaseT>(); };
      BackendStorage<typename CaseT::backend>::instance()->register_case(name, factory);
      Manager::instance()->add_preset(name, DEFAULT_PRESET, {DEFAULT_DESCRIPTION, factory()->gather_options()},
                                      ComponentType::CASE);
    }
  };

  template <class KernelT>
  class KernelRegistrar {
  public:
    explicit KernelRegistrar(const std::string &name) {
      auto factory = []() -> std::shared_ptr<ICase<typename KernelT::backend>> {
        return std::make_shared<KernelCase<KernelT>>();
      };
      BackendStorage<typename KernelT::backend>::instance()->register_case(name, factory);
      Manager::instance()->add_preset(name, DEFAULT_PRESET, {DEFAULT_DESCRIPTION, factory()->gather_options()},
                                      ComponentType::CASE);
    }
  };

  template <class StatT>
  class BackendStatRegistrar {
  public:
    explicit BackendStatRegistrar(const std::string &name) {
      BackendStorage<typename StatT::backend>::instance()->register_backend_stats(
          name, [](std::shared_ptr<Stats::StatsEngine> engine) { engine->register_stat<StatT>(); });
    }
  };

} // namespace Baseliner

#endif // BASELINER_REGISTRARS_HPP