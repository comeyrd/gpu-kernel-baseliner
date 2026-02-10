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

    protected:
      int m_buffer_size{}; // NOLINT
      int *m_l2_buffer{};  // NOLINT
    };
    class IBlockingKernel {
    public:
      virtual void block(std::shared_ptr<stream_t> stream, double timeout) = 0;
      void unblock() {
        volatile int &flag = m_host_flag;
        flag = 1;

        const volatile int &timeout_flag = m_host_timeout_flag;
        if (timeout_flag != 0) {
          IBlockingKernel::timeout_detected();
        }
      }
      virtual ~IBlockingKernel() = default;
      IBlockingKernel() = default;
      IBlockingKernel(const IBlockingKernel &) = delete;
      auto operator=(const IBlockingKernel &) -> IBlockingKernel & = delete;
      IBlockingKernel(IBlockingKernel &&) = delete;
      auto operator=(IBlockingKernel &&) -> IBlockingKernel & = delete;

    protected:
      int m_host_flag{};            // NOLINT
      int m_host_timeout_flag{};    // NOLINT
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
