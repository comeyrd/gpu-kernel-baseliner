#include <baseliner/Task.hpp>
#include <memory>
#include <vector>
namespace Baseliner {

  auto TaskManager::instance() -> TaskManager * {
    static TaskManager manager;
    return &manager;
  }
  auto TaskManager::get_tasks() -> std::vector<std::shared_ptr<ITask>> {
    return _executables;
  };

  void TaskManager::register_task(std::shared_ptr<ITask> impl) {
    _executables.push_back(std::move(impl));
  };
} // namespace Baseliner