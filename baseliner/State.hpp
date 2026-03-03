#ifndef BASELINER_STATE_HPP
#define BASELINER_STATE_HPP
#include <atomic>
#include <csignal>
#include <iostream>
namespace Baseliner {

  enum class StopState {
    Running,
    StopRequested,
    HardExitPending
  };

  class ExecutionController {
  public:
    static auto exit_requested() -> bool {
      return m_state.load() != StopState::Running;
    }

    static void handle_interrupt() {
      StopState current = m_state.load();
      if (current == StopState::Running) {
        m_state.store(StopState::StopRequested);
        std::cout << "\n[Baseliner] Stopping requested. Finishing current task...\n" << std::flush;
      } else if (current == StopState::StopRequested) {
        m_state.store(StopState::HardExitPending);
        std::cout << "\n[Baseliner] Stopping in progress. Press Ctrl+C again to hard exit.\n" << std::flush;
      } else {
        std::cerr << "\n[Baseliner] Hard exit.\n" << std::flush;
        std::exit(130);
      }
    }

  private:
    static inline std::atomic<StopState> m_state{StopState::Running};
  };

  inline void ExecutionControllerSignalHandler(int signal) {
    if (signal == SIGINT)
      ExecutionController::handle_interrupt();
  }

} // namespace Baseliner
#endif // BASELINER_STATE_HPP