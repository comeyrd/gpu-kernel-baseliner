#ifndef BASELINER_CORE_WORKLOAD_HPP
#define BASELINER_CORE_WORKLOAD_HPP
#include <baseliner/core/Options.hpp>

#include <baseliner/core/hardware/Backend.hpp>
#include <baseliner/core/stats/Stats.hpp>
#include <baseliner/core/stats/StatsEngine.hpp>
#include <memory>
#include <optional>
namespace Baseliner {
  constexpr int DEFAULT_SEED = 333;
  constexpr size_t DEFAULT_WORK_SIZE = 10;

  template <typename BackendT>
  class IWorkload : public IOption {
  public:
    using backend = BackendT;
    IWorkload() = default;
    virtual ~IWorkload() = default;
    void register_options() override {
      this->add_option("Workload", "work_size", "The work size to apply, 1 = 32MFlop & 1 = 1MB", m_work_size);
      this->add_option("Workload", "seed", "The seed used for the generation of input data", m_seed);
    }
    virtual auto number_of_floating_point_operations() -> std::optional<size_t> {
      return {};
    }
    virtual auto number_of_bytes() -> std::optional<size_t> {
      return {};
    }
    [[nodiscard]] auto get_work_size() const -> size_t {
      return m_work_size;
    }
    [[nodiscard]] auto get_seed() const -> int {
      return m_seed;
    }

    virtual auto name() -> std::string = 0;
    virtual void workload_setup_metrics(std::shared_ptr<Stats::StatsEngine> & /*engine*/) {};
    virtual void workload_update_metrics(std::shared_ptr<Stats::StatsEngine> & /*engine*/) {};
    //
    auto set_timer(std::shared_ptr<ITimer<BackendT>> timer) {
      m_timer = timer;
    }
    // Base interface

    virtual void setup(std::shared_ptr<typename BackendT::stream_t> stream) = 0;
    virtual void reset_workload(std::shared_ptr<typename BackendT::stream_t> stream) = 0;
    virtual auto run_workload(std::shared_ptr<typename BackendT::stream_t> stream) ->
        typename backend::launch_result_t = 0;
    virtual void teardown(std::shared_ptr<typename BackendT::stream_t> stream) = 0;

    // Timed Interface
    virtual auto timed_sync_setup(std::shared_ptr<typename BackendT::stream_t> stream) -> float_milliseconds {
      m_timer->init(stream);
      m_timer->measure_before(stream);
      this->setup(stream);
      m_timer->measure_after(stream);
      return m_timer->elapsed();
    };
    virtual auto timed_sync_reset_workload(std::shared_ptr<typename BackendT::stream_t> stream) -> float_milliseconds {
      m_timer->init(stream);
      m_timer->measure_before(stream);
      this->reset_workload(stream);
      m_timer->measure_after(stream);
      return m_timer->elapsed();
    };
    virtual auto timed_sync_run_workload(std::shared_ptr<typename BackendT::stream_t> stream) -> float_milliseconds {
      m_timer->init(stream);
      m_timer->measure_before(stream);
      m_timer->measure_consume(this->run_workload(stream));
      m_timer->measure_after(stream);
      return m_timer->elapsed();
    };
    virtual void init_batch(std::shared_ptr<typename BackendT::stream_t> stream, size_t batch_size, bool is_blocking) {
      m_timer->init_batch(stream, batch_size, is_blocking);
    }
    virtual void timed_batch_run_workload(std::shared_ptr<typename BackendT::stream_t> stream) {
      m_timer->measure_batch_before(stream);
      m_timer->measure_batch_consume(this->run_workload(stream));
      m_timer->measure_batch_after(stream);
    };
    virtual auto timed_run_elapsed_batch() -> std::vector<float_milliseconds> {
      return m_timer->elapsed_batch();
    }
    virtual auto timed_sync_teardown(std::shared_ptr<typename BackendT::stream_t> stream) -> float_milliseconds {
      m_timer->init(stream);
      m_timer->measure_before(stream);
      this->teardown(stream);
      m_timer->measure_after(stream);
      return m_timer->elapsed();
    };

    //
    virtual auto validate_workload() -> bool = 0;

    // Metrics management
    void setup_metrics(std::shared_ptr<Stats::StatsEngine> engine) {
      std::optional<size_t> bytes = this->number_of_bytes();
      std::optional<size_t> flops = this->number_of_floating_point_operations();
      if (bytes.has_value()) {
        engine->register_metric<Stats::ByteNumbers>(bytes.value());
        engine->register_stat<Stats::DataTroughput>();
        m_bytes = true;
      }
      if (flops.has_value()) {
        engine->register_metric<Stats::FLOPCount>(flops.value());
        engine->register_stat<Stats::FLOPThroughputaTroughput>();
        m_flops = true;
      }
      if (m_bytes && m_flops) {
        engine->register_stat<Stats::ArithmeticIntensity>();
      }
      this->workload_setup_metrics(engine);
    };
    void update_metrics(std::shared_ptr<Stats::StatsEngine> engine) {
      std::optional<size_t> bytes = this->number_of_bytes();
      std::optional<size_t> flops = this->number_of_floating_point_operations();
      if (m_bytes) {
        engine->update_values<Stats::ByteNumbers>(bytes.value());
      }
      if (m_flops) {
        engine->update_values<Stats::FLOPCount>(flops.value());
      }
      this->workload_update_metrics(engine);
    };

  private:
    bool m_bytes = false;
    bool m_flops = false;
    std::shared_ptr<ITimer<BackendT>> m_timer;
    size_t m_work_size = DEFAULT_WORK_SIZE;
    int m_seed = DEFAULT_SEED;
  };

} // namespace Baseliner
#endif // BASELINER_WORKLOAD_HPP