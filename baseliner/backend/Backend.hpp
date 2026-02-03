#ifndef BACKEND_HPP
#define BACKEND_HPP
#include <baseliner/Timer.hpp>
#include <iostream>
#include <memory>
namespace Baseliner {
  namespace Backend {
    template <typename E, typename S>
    class IDevice {
    public:
      using event_t = E;
      using stream_t = S;
      virtual std::shared_ptr<stream_t> create_stream() = 0;
      virtual void synchronize(std::shared_ptr<stream_t> stream) = 0;
      virtual void get_last_error() = 0;
      virtual void set_device(int device) = 0;
      virtual void reset_device() = 0;
      virtual ~IDevice() = default;

      class L2Flusher {
      public:
        virtual void flush(std::shared_ptr<stream_t> stream) = 0;
        virtual ~L2Flusher() = default;

      protected:
        int m_buffer_size{};
        int *m_l2_buffer{};
      };
      class BlockingKernel {
      public:
        virtual void block(std::shared_ptr<stream_t> stream, double timeout) = 0;
        inline void unblock() {
          volatile int &flag = m_host_flag;
          flag = 1;

          const volatile int &timeout_flag = m_host_timeout_flag;
          if (timeout_flag) {
            BlockingKernel::timeout_detected();
          }
        }
        virtual ~BlockingKernel() = default;

      protected:
        int m_host_flag{};
        int m_host_timeout_flag{};
        int *m_device_flag{};
        int *m_device_timeout_flag{};

        static void timeout_detected() {
          std::cout << "Deadlock detected" << std::endl;
        };
      };
      class Timer : public IGpuTimer<stream_t> {};
    };
  } // namespace Backend
} // namespace Baseliner

#endif // BACKEND_HPP
