#ifndef BASELINER_EXECUTABLE_HPP
#define BASELINER_EXECUTABLE_HPP
#include <baseliner/Result.hpp>
#include <vector>
namespace Baseliner {
  class Executable {
  public:
    virtual std::vector<Result> run_all() = 0;
    virtual ~Executable() = default;
  };
  class SingleExecutable : public Executable {
  public:
    virtual Result run() = 0;
    std::vector<Result> run_all() override {
      return std::vector{run()};
    };
  };

  class ExecutableManager {
  private:
    std::vector<std::shared_ptr<Executable>> _executables;

  public:
    static ExecutableManager *instance();
    const std::vector<std::shared_ptr<Executable>> &getExecutables();

    void register_executable(std::shared_ptr<Executable> impl);
  };

  class RegistrarExecutable {
  public:
    RegistrarExecutable(Executable *impl) {
      // Create a shared_ptr with a custom lambda deleter that does NOTHING
      auto skip_delete = [](Executable *) { /* do nothing */ };

      ExecutableManager::instance()->register_executable(std::shared_ptr<Executable>(impl, skip_delete));
    }
    template <typename T>
    RegistrarExecutable(std::shared_ptr<T> impl) {
      ExecutableManager::instance()->register_executable(std::shared_ptr<Executable>(impl));
    }
  };

} // namespace Baseliner
#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif
#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define BASELINER_REGISTER_EXECUTABLE(item)                                                                            \
  static Baseliner::RegistrarExecutable ATTRIBUTE_USED CONCAT(registrar_, __LINE__)(item);

#endif // BASELINER_EXECUTABLE_HPP
