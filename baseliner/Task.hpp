#ifndef BASELINER_TASK_HPP
#define BASELINER_TASK_HPP
#include <baseliner/Result.hpp>
#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>
namespace Baseliner {

  class ITask {
  public:
    virtual auto name() -> std::string = 0;
    virtual auto run_all() -> std::vector<Result> = 0;
    virtual auto print_console(const std::vector<Result> &results) -> std::string {
      return "";
    };
    virtual ~ITask() = default;
  };
  class ISingleTask : public ITask {
  public:
    virtual auto run() -> Result = 0;
    auto run_all() -> std::vector<Result> override {
      return std::vector{run()};
    };
  };

  class TaskManager {
  private:
    std::vector<std::shared_ptr<ITask>> _executables;

  public:
    static auto instance() -> TaskManager *;
    auto get_tasks() -> std::vector<std::shared_ptr<ITask>>;

    void register_task(std::shared_ptr<ITask> impl);
  };

  class TaskRegistrar {
  public:
    TaskRegistrar(ITask *impl) {
      // Create a shared_ptr with a custom lambda deleter that does NOTHING
      auto skip_delete = [](ITask *) { /* do nothing */ };

      TaskManager::instance()->register_task(std::shared_ptr<ITask>(impl, skip_delete));
    }
    template <typename T>
    TaskRegistrar(std::shared_ptr<T> impl) {
      TaskManager::instance()->register_task(std::shared_ptr<ITask>(impl));
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
#define BASELINER_REGISTER_TASK(item)                                                                                  \
  static const Baseliner::TaskRegistrar ATTRIBUTE_USED CONCAT(registrar_, __LINE__)(item);

#endif // BASELINER_TASK_HPP
