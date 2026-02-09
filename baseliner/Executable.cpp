#include <baseliner/Executable.hpp>
namespace Baseliner {

  ExecutableManager *ExecutableManager::instance() {
    static ExecutableManager manager;
    return &manager;
  }
  const std::vector<std::shared_ptr<Executable>> &ExecutableManager::getExecutables() {
    return _executables;
  };

  void ExecutableManager::register_executable(std::shared_ptr<Executable> impl) {
    _executables.push_back(impl);
  };
} // namespace Baseliner