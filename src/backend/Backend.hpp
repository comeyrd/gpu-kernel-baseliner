#ifndef BACKEND_HPP
#define BACKEND_HPP
#include "ITimer.hpp"
#include <iostream>
namespace Baseliner {
  namespace Backend {
    template <typename event_t, typename stream_t>
    class IDevice {
    public:
      virtual void synchronize(stream_t stream) = 0;
      virtual void set_device(int device) = 0;
      virtual void reset_device() = 0;
      class L2Flusher {
      public:
        virtual void flush(stream_t stream) = 0;

      protected:
        int buffer_size{};
        int *l2_buffer{};
      };
      class BlockingKernel {
      public:
        virtual void block(stream_t stream, double timeout) = 0;
        inline void unblock() {
          volatile int &flag = m_host_flag;
          flag = 1;

          const volatile int &timeout_flag = m_host_timeout_flag;
          if (timeout_flag) {
            BlockingKernel::timeout_detected();
          }
        }

      protected:
        int m_host_flag{};
        int m_host_timeout_flag{};
        int *m_device_flag{};
        int *m_device_timeout_flag{};

        static void timeout_detected() {
          std::cout << "Deadlock detected" << std::endl;
        };
      };
      class GpuTimer : public ITimer<stream_t> {
      private:
        event_t start_event;
        event_t stop_event;
      };
    };
  } // namespace Backend
} // namespace Baseliner

#endif // BACKEND_HPP
