#ifndef BASELINER_BACKEND_MANAGER
#define BASELINER_BACKEND_MANAGER
#include <baseliner/Benchmark.hpp>
#include <baseliner/Case.hpp>
#include <baseliner/managers/BenchmarkCaseManager.hpp>

namespace Baseliner {

  class BackendManager {
  public:
    static void register_backend(const std::string &name, IBenchmarkCaseManager *manager) {
      get_backends()[name] = manager;
    }

    static IBenchmarkCaseManager *get(const std::string &name) {
      auto &backends = get_backends();
      if (backends.count(name) > 0) {
        return backends[name];
      }
      throw std::runtime_error("Backend not available/compiled: " + name);
    }
    static auto list_backends() -> std::vector<std::string> {
      std::vector<std::string> vecstr;
      for (const auto &[name, _] : get_backends()) {
        vecstr.push_back(name);
      }
      return vecstr;
    }

  private:
    static auto get_backends() -> std::unordered_map<std::string, IBenchmarkCaseManager *> & {
      static std::unordered_map<std::string, IBenchmarkCaseManager *> backends;
      return backends;
    }
  };

  template <class BackendT>
  class BackendRegistrar {
  public:
    explicit BackendRegistrar(std::string name) {
      auto instance = BenchmarkCaseManager<BackendT>::instance();
      BackendManager::register_backend(name, instance);
    }
  };

} // namespace Baseliner

#define BASELINER_REGISTER_BACKEND(BackendClass, name)                                                                 \
  ATTRIBUTE_USED static Baseliner::BackendRegistrar<BackendClass> _registrar_##BackendClass{name};
#endif // BASELINER_BACKEND_MANAGER