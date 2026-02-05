#ifndef RUNNER_HPP
#define RUNNER_HPP
#include <baseliner/Kernel.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <baseliner/Timer.hpp>
#include <baseliner/backend/Backend.hpp>
#include <iostream>
namespace Baseliner {

  class RunnerBase : public OptionConsumer, public OptionBroadcaster {
  public:
    // Runner Options
    bool m_warmup = true;
    bool m_flush_l2 = true;
    bool m_block = false;
    float m_block_duration_ms = 1000.0;
    // OptionConsumer Interfac
    virtual Result run() = 0;
    explicit RunnerBase(IStoppingCriterion &stopping)
        : m_stopping(stopping) {};

    virtual ~RunnerBase() = default;

    void register_options() override {
      add_option("Runner", "block", "Using a blocking kernel", m_block);
      add_option("Runner", "block_duration", "Duration of the blocking kernel (in ms)", m_block_duration_ms);
      add_option("Runner", "flush", "Enables the flushing of the L2 cache", m_flush_l2);
      add_option("Runner", "warmup", "Having a warmup run", m_warmup);
    }

  protected:
    IStoppingCriterion &m_stopping;
  };
  template <typename Kernel, typename Device>
  // TODO Setup the static checks at compile time.
  class Runner : public RunnerBase {
  public:
    // OptionBroadcaster
    void register_dependencies() override {
      register_consumer(m_input);
      register_consumer(m_stopping);
      register_consumer(*this);
    };
    // Runner
    explicit Runner(IStoppingCriterion &stopping)
        : RunnerBase(stopping),
          m_input(),
          m_backend(),
          m_stream(m_backend.create_stream()),
          m_flusher(),
          m_blocker(),
          m_kernel(std::make_unique<Kernel>(m_input)) {};

    Result run() override {
      m_input.generate_random();
      typename Kernel::Output m_out_cpu(m_input);
      typename Kernel::Output m_out_gpu(m_input);
      m_kernel->cpu(m_out_cpu);
      m_kernel->setup();
      preAll();
      while (!m_stopping.satisfied()) {
        m_kernel->reset();
        preRun();
        m_kernel->timed_run(m_stream);
        postRun();
        m_stopping.addTime(m_kernel->time_elapsed());
      }
      postAll();
      m_kernel->teardown(m_out_gpu);
      if (!(m_out_gpu == m_out_cpu)) {
        std::cout << "Error, GPU and CPU results are not the same" << std::endl;
        // std::cout << m_out_gpu << std::endl << " | " << m_out_cpu << std::endl;
      }

      // TODO Fix git version ?
      Result result(this->gather_options(), m_kernel->name(), "");
      std::vector<Metric> metrics = m_stopping.get_metrics();
      result.push_back_metrics(metrics);
      return result;
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
      m_backend.get_last_error();
      if (m_block)
        m_blocker.unblock();
    };
    virtual void postAll() {
      m_backend.synchronize(m_stream);
    };
  };
} // namespace Baseliner

#endif // RUNNER_HPP