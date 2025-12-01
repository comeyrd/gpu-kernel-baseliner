#ifndef RUNNER_HPP
#define RUNNER_HPP
#include "ITimer.hpp"
#include "Kernel.hpp"
#include "StoppingCriterion.hpp"
#include "backend/Backend.hpp"
#include <iostream>
namespace Baseliner {

  template <typename Kernel, typename Device>
  // TODO Setup the static checks at compile time.
  class Runner : public OptionConsumer, public OptionBroadcaster {
  public:
    // Runner Options
    bool m_warmup = true;
    bool m_flush_l2 = true;
    bool m_block = false;
    float m_block_duration_ms = 1000.0;

    // OptionConsumer Interface
    const std::string get_name() override {
      return "Runner";
    }

    void register_options() override {
      add_option("block", "Using a blocking kernel", m_block);
      add_option("block_duration", "Duration of the blocking kernel (in ms)", m_block_duration_ms);
      add_option("flush", "Enables the flushing of the L2 cache", m_flush_l2);
      add_option("warmup", "Having a warmup run", m_warmup);
    }
    // OptionBroadcaster
    void register_dependencies() override {
      register_consumer(m_input);
    };
    // Runner
    // TODO delay the instantiation of everything, because options...
    explicit Runner(IStoppingCriterion &stopping)
        : m_stopping(stopping),
          m_input(),
          m_backend(),
          m_stream(m_backend.create_stream()),
          m_flusher(),
          m_timer(std::make_unique<typename Device::GpuTimer>(m_stream)),
          m_blocker(),
          m_kernel(std::make_unique<Kernel>(m_input)) {};
    std::vector<float_milliseconds> run() {
      m_input.generate_random();
      typename Kernel::Output m_out_cpu(m_input);
      typename Kernel::Output m_out_gpu(m_input);
      m_kernel->cpu(m_out_cpu);
      m_kernel->setup();
      preAll();
      while (!m_stopping.satisfied()) {
        m_kernel->reset();
        preRun();
        m_timer->start();
        m_kernel->run(m_stream);
        m_timer->stop();
        postRun();
        m_stopping.addTime(m_timer->time_elapsed());
      }
      postAll();
      m_kernel->teardown(m_out_gpu);
      if (!(m_out_gpu == m_out_cpu)) {
        std::cout << "Cpu and Gpu cooked different results" << std::endl;
      }
      return m_stopping.executionTimes();
    }

  protected:
    // Kernel Types
    typename Kernel::Input m_input;

    // Backend specifics
    Device m_backend;
    typename Device::L2Flusher m_flusher;
    typename Device::BlockingKernel m_blocker;
    std::shared_ptr<typename Device::stream_t> m_stream;

    // Kernel Holder
    std::unique_ptr<Kernel> m_kernel;

    // StoppingCriterion and Timer
    IStoppingCriterion &m_stopping;
    std::unique_ptr<ITimer> m_timer;

    virtual void preAll() {
      if (m_warmup)
        m_kernel->run(m_stream);
      m_stopping.reset();
    };
    virtual void preRun() {
      if (m_flush_l2)
        m_flusher.flush(m_stream);
      if (m_block)
        m_blocker.block(m_stream, m_block_duration_ms);
    };
    virtual void postRun() {
      if (m_block)
        m_blocker.unblock();
    };
    virtual void postAll() {};
  };
} // namespace Baseliner

#endif // RUNNER_HPP