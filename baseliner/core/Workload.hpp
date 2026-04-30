#ifndef BASELINER_CORE_WORKLOAD_HPP
#define BASELINER_CORE_WORKLOAD_HPP
#include <baseliner/core/Options.hpp>
#include <baseliner/core/Timer.hpp>
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

    void register_options() override;

    virtual auto number_of_floating_point_operations() -> std::optional<size_t>;
    virtual auto number_of_bytes() -> std::optional<size_t>;

    [[nodiscard]] auto get_work_size() const -> size_t;
    [[nodiscard]] auto get_seed() const -> int;
    virtual auto algo() -> std::string;
    virtual auto specialization() -> std::string;

    auto set_timer(std::shared_ptr<ITimer<BackendT>> timer);

    // Base interface
    virtual void setup_host() {
      if (m_validation_type == "File") {
        setup_host_from_file(m_path);
      } else {
        setup_host_random_generated();
        if (m_validation_type == "Reference") {
          results_from_reference();
        }
      }
    };
    virtual void setup_device(typename BackendT::stream_t stream) = 0;
    virtual void reset_device(typename BackendT::stream_t stream) = 0;
    virtual auto run(typename BackendT::stream_t stream) -> typename backend::launch_result_t = 0;
    virtual void fetch_results(typename BackendT::stream_t stream) = 0;
    virtual void free() = 0;
    virtual auto validate() -> bool {
      return false;
    };

    // Timed Interface
    virtual auto timed_sync_setup_device(typename BackendT::stream_t stream) -> float_milliseconds;
    virtual auto timed_sync_reset_device(typename BackendT::stream_t stream) -> float_milliseconds;
    virtual auto timed_sync_run(typename BackendT::stream_t stream) -> float_milliseconds;
    virtual auto timed_sync_fetch_results(typename BackendT::stream_t stream) -> float_milliseconds;
    virtual auto timed_sync_free(typename BackendT::stream_t stream) -> float_milliseconds;

    // Batch management
    virtual void init_batch(typename BackendT::stream_t stream, size_t batch_size, bool is_blocking);
    virtual void timed_batch_run(typename BackendT::stream_t stream);
    virtual auto timed_run_elapsed_batch() -> std::vector<float_milliseconds>;

    // Metrics management
    void setup_metrics(std::shared_ptr<Stats::StatsEngine> engine);
    void update_metrics(std::shared_ptr<Stats::StatsEngine> engine);
    virtual void inner_setup_metrics(std::shared_ptr<Stats::StatsEngine> engine);
    virtual void inner_update_metrics(std::shared_ptr<Stats::StatsEngine> engine);

    // Validation & saving 2 file
    virtual void setup_host_from_file(std::string & /*path*/) {};
    virtual void setup_host_random_generated() {};
    virtual void save_setup();
    virtual void inner_save_setup(std::string & /*path*/) {};
    virtual void results_from_reference() {};

  private:
    bool m_bytes = false;
    bool m_flops = false;
    std::shared_ptr<ITimer<BackendT>> m_timer;
    size_t m_work_size = DEFAULT_WORK_SIZE;
    int m_seed = DEFAULT_SEED;
    std::string m_path;
    std::string m_validation_type = "Reference";
  };

  // IMPLEMENTATION
  template <typename BackendT>
  void IWorkload<BackendT>::register_options() {
    this->add_option("Workload", "work_size", "The work size to apply, 1 = 32MFlop & 1 = 1MB", m_work_size);
    this->add_option("Workload", "seed", "The seed used for the generation of input data", m_seed);
    this->add_option("Workload", "validation_path",
                     "if the validation is file based, the path where the comparison files is located", m_path);
    this->add_option("Workload", "validation_type", "How the workload should be validated", m_validation_type);
  }

  template <typename BackendT>
  void IWorkload<BackendT>::save_setup() {
    inner_save_setup(m_path);
  }

  template <typename BackendT>
  auto IWorkload<BackendT>::number_of_floating_point_operations() -> std::optional<size_t> {
    return {};
  }

  template <typename BackendT>
  auto IWorkload<BackendT>::number_of_bytes() -> std::optional<size_t> {
    return {};
  }

  template <typename BackendT>
  auto IWorkload<BackendT>::get_work_size() const -> size_t {
    return m_work_size;
  }

  template <typename BackendT>
  auto IWorkload<BackendT>::get_seed() const -> int {
    return m_seed;
  }

  template <typename BackendT>
  auto IWorkload<BackendT>::algo() -> std::string {
    return {};
  }

  template <typename BackendT>
  auto IWorkload<BackendT>::specialization() -> std::string {
    return {};
  }

  template <typename BackendT>
  void IWorkload<BackendT>::inner_setup_metrics(std::shared_ptr<Stats::StatsEngine> /*engine*/) {
  }

  template <typename BackendT>
  void IWorkload<BackendT>::inner_update_metrics(std::shared_ptr<Stats::StatsEngine> /*engine*/) {
  }

  template <typename BackendT>
  auto IWorkload<BackendT>::set_timer(std::shared_ptr<ITimer<BackendT>> timer) {
    m_timer = timer;
  }

  // Timed Interface

  template <typename BackendT>
  auto IWorkload<BackendT>::timed_sync_setup_device(typename BackendT::stream_t stream) -> float_milliseconds {
    m_timer->init(stream);
    m_timer->measure_before(stream);
    this->setup_device(stream);
    m_timer->measure_after(stream);
    return m_timer->elapsed();
  }

  template <typename BackendT>
  auto IWorkload<BackendT>::timed_sync_reset_device(typename BackendT::stream_t stream) -> float_milliseconds {
    m_timer->init(stream);
    m_timer->measure_before(stream);
    this->reset_device(stream);
    m_timer->measure_after(stream);
    return m_timer->elapsed();
  }

  template <typename BackendT>
  auto IWorkload<BackendT>::timed_sync_run(typename BackendT::stream_t stream) -> float_milliseconds {
    m_timer->init(stream);
    m_timer->measure_before(stream);
    m_timer->measure_consume(this->run(stream));
    m_timer->measure_after(stream);
    return m_timer->elapsed();
  }

  template <typename BackendT>
  void IWorkload<BackendT>::init_batch(typename BackendT::stream_t stream, size_t batch_size, bool is_blocking) {
    m_timer->init_batch(stream, batch_size, is_blocking);
  }

  template <typename BackendT>
  void IWorkload<BackendT>::timed_batch_run(typename BackendT::stream_t stream) {
    m_timer->measure_batch_before(stream);
    m_timer->measure_batch_consume(this->run(stream));
    m_timer->measure_batch_after(stream);
  }

  template <typename BackendT>
  auto IWorkload<BackendT>::timed_run_elapsed_batch() -> std::vector<float_milliseconds> {
    return m_timer->elapsed_batch();
  }

  template <typename BackendT>
  auto IWorkload<BackendT>::timed_sync_fetch_results(typename BackendT::stream_t stream) -> float_milliseconds {
    m_timer->init(stream);
    m_timer->measure_before(stream);
    this->fetch_results(stream);
    m_timer->measure_after(stream);
    return m_timer->elapsed();
  }

  template <typename BackendT>
  auto IWorkload<BackendT>::timed_sync_free(typename BackendT::stream_t stream) -> float_milliseconds {
    m_timer->init(stream);
    m_timer->measure_before(stream);
    this->free();
    m_timer->measure_after(stream);
    return m_timer->elapsed();
  }

  // Metrics management

  template <typename BackendT>
  void IWorkload<BackendT>::setup_metrics(std::shared_ptr<Stats::StatsEngine> engine) {
    std::optional<size_t> bytes = this->number_of_bytes();
    std::optional<size_t> flops = this->number_of_floating_point_operations();
    if (bytes.has_value()) {
      engine->register_metric<Stats::ByteNumbers>(bytes.value());
      engine->register_stat<Stats::DataThroughput>();
      m_bytes = true;
    }
    if (flops.has_value()) {
      engine->register_metric<Stats::FLOPCount>(flops.value());
      engine->register_stat<Stats::FLOPThroughput>();
      m_flops = true;
    }
    if (m_bytes && m_flops) {
      engine->register_stat<Stats::ArithmeticIntensity>();
    }
    this->inner_setup_metrics(engine);
  }

  template <typename BackendT>
  void IWorkload<BackendT>::update_metrics(std::shared_ptr<Stats::StatsEngine> engine) {
    std::optional<size_t> bytes = this->number_of_bytes();
    std::optional<size_t> flops = this->number_of_floating_point_operations();
    if (m_bytes) {
      engine->update_values<Stats::ByteNumbers>(bytes.value());
    }
    if (m_flops) {
      engine->update_values<Stats::FLOPCount>(flops.value());
    }
    this->inner_update_metrics(engine);
  }

} // namespace Baseliner
#endif // BASELINER_WORKLOAD_HPP