#ifndef BASELINER_SUITE_MANAGER
#define BASELINER_SUITE_MANAGER
#include <baseliner/Axe.hpp>
#include <baseliner/Benchmark.hpp>
#include <baseliner/Case.hpp>
namespace Baseliner {

  class SuiteManager {
  public:
    static void register_suite(const std::string &name, const Axe &axe) {
      get_suites()[name] = axe;
    }
    static auto list_suites() -> std::vector<std::string> {
      std::vector<std::string> vecstr;
      for (const auto &[name, _] : get_suites()) {
        vecstr.push_back(name);
      }
      return vecstr;
    }
    static auto get_suite(const std::string &name) {
      auto &suite = get_suites();
      if (suite.count(name) > 0) {
        return suite[name];
      }
      throw std::runtime_error("Suite not available/compiled: " + name);
    }

  private:
    static auto get_suites() -> std::unordered_map<std::string, Axe> & {
      static std::unordered_map<std::string, Axe> suites;
      return suites;
    }
  };

  class SuiteRegistrar {
  public:
    explicit SuiteRegistrar(const Axe &axe) {
      SuiteManager::register_suite(axe.axe_name, axe);
    }
  };

} // namespace Baseliner

#define BASELINER_REGISTER_SUITE(Axe) ATTRIBUTE_USED static Baseliner::SuiteRegistrar _registrar_##Axe{Axe};
#endif // BASELINER_SUITE_MANAGER