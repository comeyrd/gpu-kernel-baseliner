#include <baseliner/Executable.hpp>
#include <memory>
#include <vector>
namespace Baseliner {

  auto ExecutableManager::instance() -> ExecutableManager * {
    static ExecutableManager manager;
    return &manager;
  }
  auto ExecutableManager::getExecutables() -> std::vector<std::shared_ptr<IExecutable>> {
    return _executables;
  };

  void ExecutableManager::register_executable(std::shared_ptr<IExecutable> impl) {
    _executables.push_back(impl);
  };
} // namespace Baseliner