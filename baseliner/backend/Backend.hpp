#ifndef BACKEND_HPP
#define BACKEND_HPP
#include <baseliner/Timer.hpp>
#include <iostream>
#include <memory>
namespace Baseliner::Backend {
  template <typename S>
  class IDevice {
  public:
    using stream_t = S;
    virtual auto create_stream() -> std::shared_ptr<stream_t> = 0;
    virtual void synchronize(std::shared_ptr<stream_t> stream) = 0;
    virtual void get_last_error() = 0;
    virtual void set_device(int device) = 0;
    virtual void reset_device() = 0;
    virtual ~IDevice() = default;

    class IL2Flusher {
    public:
      virtual void flush(std::shared_ptr<stream_t> stream) = 0;
      virtual ~IL2Flusher() = default;
      IL2Flusher() = default;
      IL2Flusher(IL2Flusher &&other) noexcept
          : m_buffer_size(other.m_buffer_size),
            m_l2_buffer(other.m_l2_buffer) {
        other.m_l2_buffer = nullptr;
      }

      auto operator=(IL2Flusher &&other) noexcept -> IL2Flusher & {
        if (this != &other) {
          m_buffer_size = other.m_buffer_size;
          m_l2_buffer = other.m_l2_buffer;

          other.m_l2_buffer = nullptr;
        }
        return *this;
      }

    protected:
      int m_buffer_size{}; // NOLINT
      int *m_l2_buffer{};  // NOLINT
    };
    class IBlockingKernel {
    public:
      virtual void block(std::shared_ptr<stream_t> stream, double timeout) = 0;
      void unblock() {
        if (m_host_flag == nullptr || m_host_timeout_flag == nullptr) {
          return;
        }
        *static_cast<volatile int *>(m_host_flag) = 1;

        if (*static_cast<volatile int *>(m_host_timeout_flag) != 0) {
          IBlockingKernel::timeout_detected();
        }
      }
      virtual ~IBlockingKernel() {
        delete m_host_flag;
        delete m_host_timeout_flag;
      };
      IBlockingKernel() {
        m_host_flag = new int;         // NOLINT
        m_host_timeout_flag = new int; // NOLINT
      };
      IBlockingKernel(IBlockingKernel &&other) noexcept
          : m_host_flag(other.m_host_flag),
            m_host_timeout_flag(other.m_host_timeout_flag),
            m_device_flag(other.m_device_flag),
            m_device_timeout_flag(other.m_device_timeout_flag) {
        other.m_device_flag = nullptr;
        other.m_device_timeout_flag = nullptr;
        other.m_host_flag = nullptr;
        other.m_host_timeout_flag = nullptr;
      }

      auto operator=(IBlockingKernel &&other) noexcept -> IBlockingKernel & {
        if (this != &other) {
          delete m_host_flag;
          delete m_host_timeout_flag;
          m_host_flag = other.m_host_flag;
          m_host_timeout_flag = other.m_host_timeout_flag;
          m_device_flag = other.m_device_flag;
          m_device_timeout_flag = other.m_device_timeout_flag;
          other.m_device_flag = nullptr;
          other.m_device_timeout_flag = nullptr;
          other.m_host_flag = nullptr;
          other.m_host_timeout_flag = nullptr;
        }
        return *this;
      }
      IBlockingKernel(const IBlockingKernel &) = delete;
      auto operator=(const IBlockingKernel &) -> IBlockingKernel & = delete;

    protected:
      int *m_host_flag;             // NOLINT
      int *m_host_timeout_flag;     // NOLINT
      int *m_device_flag{};         // NOLINT
      int *m_device_timeout_flag{}; // NOLINT

      static void timeout_detected() {
        std::cout << "Deadlock detected" << "\n";
      };
    };
    class ITimer : public IGpuTimer<stream_t> {};
  };
} // namespace Baseliner::Backend

#endif // BACKEND_HPP
