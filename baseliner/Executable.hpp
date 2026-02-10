#ifndef BASELINER_EXECUTABLE_HPP
#define BASELINER_EXECUTABLE_HPP
#include <baseliner/Result.hpp>
#include <memory>
#include <vector>
namespace Baseliner {
  class IExecutable {
  public:
    virtual auto run_all() -> std::vector<Result> = 0;
    virtual ~IExecutable() = default;
  };
  class ISingleExecutable : public IExecutable {
  public:
    virtual auto run() -> Result = 0;
    auto run_all() -> std::vector<Result> override {
      return std::vector{run()};
    };
  };

  class ExecutableManager {
  private:
    std::vector<std::shared_ptr<IExecutable>> _executables;

  public:
    static auto instance() -> ExecutableManager *;
    auto getExecutables() -> std::vector<std::shared_ptr<IExecutable>>;

    void register_executable(std::shared_ptr<IExecutable> impl);
  };

  class RegistrarExecutable {
  public:
    RegistrarExecutable(IExecutable *impl) {
      // Create a shared_ptr with a custom lambda deleter that does NOTHING
      auto skip_delete = [](IExecutable *) { /* do nothing */ };

      ExecutableManager::instance()->register_executable(std::shared_ptr<IExecutable>(impl, skip_delete));
    }
    template <typename T>
    RegistrarExecutable(std::shared_ptr<T> impl) {
      ExecutableManager::instance()->register_executable(std::shared_ptr<IExecutable>(impl));
    }
  };

} // namespace Baseliner
#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif
#define CONCAT_IMPL(a, b) a##b
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CONCAT(a, b) CONCAT_IMPL(a, b)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define BASELINER_REGISTER_EXECUTABLE(item)                                                                            \
  static Baseliner::RegistrarExecutable ATTRIBUTE_USED CONCAT(registrar_, __LINE__)(item);

#endif // BASELINER_EXECUTABLE_HPP
