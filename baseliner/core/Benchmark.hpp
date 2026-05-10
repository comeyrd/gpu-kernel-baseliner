#ifndef BASELINER_CORE_BENCHMARK_HPP
#define BASELINER_CORE_BENCHMARK_HPP
#include <baseliner/core/BenchmarkReport.hpp>
#include <baseliner/core/Error.hpp>

#include <baseliner/core/IPrinter.hpp>
#include <baseliner/core/Metric.hpp>
#include <baseliner/core/OptionTypes.hpp>

#include <baseliner/core/Options.hpp>

#include <baseliner/core/State.hpp>
#include <baseliner/core/StoppingCriterion.hpp>
#include <baseliner/core/Workload.hpp>
#include <baseliner/core/hardware/Backend.hpp>
#include <baseliner/core/hardware/BackendStats.hpp>
#include <baseliner/core/stats/IStats.hpp>
#include <baseliner/core/stats/Stats.hpp>
#include <baseliner/core/stats/StatsEngine.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
inline static const std::string_view DEFAULT_BENCHMARK_NAME = "Benchmark";

namespace Baseliner {
  constexpr float DEFAULT_BLOCK_DURATION = 1000.0F;

  class IBenchmark : public IOption {
  public:
    // Benchmark Options
    // IOption Interface
    explicit IBenchmark()
        : m_stats_engine(std::make_shared<Stats::StatsEngine>()) {};
    virtual auto name() -> std::string = 0;

    IBenchmark(IBenchmark &&) noexcept = default;
    auto operator=(IBenchmark &&) noexcept -> IBenchmark & = default;

    IBenchmark(const IBenchmark &) = delete;
    auto operator=(const IBenchmark &) -> IBenchmark & = delete;

    ~IBenchmark() override = default;
    virtual auto run_benchmark() -> BenchmarkReport = 0;

    void set_m_warmup(bool warmup) {
      m_warmup = warmup;
    };
    [[nodiscard]] auto get_warmup() const -> bool {
      return m_warmup;
    };
    void set_m_flush_l2(bool flush_l2) {
      m_flush_l2 = flush_l2;
    };
    [[nodiscard]] auto get_flush_l2() const -> bool {
      return m_flush_l2;
    };
    void set_m_block(bool block) {
      m_block = block;
    };
    [[nodiscard]] auto get_block() const -> bool {
      return m_block;
    };
    [[nodiscard]] auto get_validate_workload() const -> bool {
      return m_validate_workload;
    };
    void set_m_block_duration(float block_duration) {
      m_block_duration_ms = block_duration;
    };
    [[nodiscard]] auto get_block_duration() const -> float {
      return m_block_duration_ms;
    };

    [[nodiscard]] auto get_first() const -> bool {
      return m_first;
    }
    void set_m_first(bool first) {
      m_first = first;
    }
    [[nodiscard]] auto get_m_name() const -> std::string {
      return m_name;
    }
    void set_m_name(std::string name) {
      m_name = std::move(name);
    }

    virtual auto get_workload_options() -> OptionsMap = 0;

    void set_stopping_criterion(const std::function<std::unique_ptr<StoppingCriterion>()> &stopping_builder) {
      m_stopping = stopping_builder();
      m_stopping->set_stats_engine(m_stats_engine);
    }
    void add_stats(const std::vector<std::function<void(std::shared_ptr<Stats::StatsEngine>)>> &stats_recipes) {
      for (auto stat : stats_recipes) {
        add_stat(stat);
      }
    }
    void add_stat(const std::function<void(std::shared_ptr<Stats::StatsEngine>)> &stat_recipe) {
      stat_recipe(m_stats_engine);
    }

    void set_stat_options(const OptionsMap &omap) {
      stats_options = omap;
    }
    [[nodiscard]] auto get_stat_options() const -> OptionsMap {
      return stats_options;
    }
    [[nodiscard]] auto get_sweep_spec() const -> std::optional<SweepSpec> {
      return m_sweep_spec;
    }
    void set_sweep_spec(const std::optional<SweepSpec> &spec) {
      m_sweep_spec = spec;
    }
    void set_backend_options(const OptionsMap &omap) {
      m_backend_options = omap;
    }
    void set_printer(std::shared_ptr<IBenchmarkPrinter> printer) {
      m_printer = printer;
    }

  protected:
    void register_options() override {
      add_option("Benchmark", "block", "Using a blocking kernel", m_block);
      add_option("Benchmark", "block_duration", "Duration of the blocking kernel (in ms)", m_block_duration_ms);
      add_option("Benchmark", "block_queue_size", "The max size of the blocking queue", m_max_blocked_queue);
      add_option("Benchmark", "flush", "Enables the flushing of the L2 cache", m_flush_l2);
      add_option("Benchmark", "warmup", "Having a warmup run", m_warmup);
      add_option("Benchmark", "batch_size", "The size of the batch", m_batch_size);
      add_option("Benchmark", "warm_cool", "Does the Benchmark actively warm or cool the GPU", m_warm_cool);
      add_option("Benchmark", "warm_cool_timeout", "How long the benchmark warms or cool before throwing",
                 m_warm_cool_timeout);
      add_option("Benchmark", "min_gpu_temp",
                 "If warm_cool, the minimum accepted temperature before warming up the GPU", m_min_gpu_temp);
      add_option("Benchmark", "max_gpu_temp",
                 "If warm_cool, the minimum accepted temperature before cooling down the GPU", m_max_gpu_temp);
      add_option("Benchmark", "minimal_batch_duration",
                 "If dynamic batch size is set tu true, how long a batch should minimaly be", m_minimal_batch_duration);
      add_option("Benchmark", "dynamic_batch", "If the batch size is dynamic", m_dynamic_batch);
      add_option("Benchmark", "validate_workload", "If you want the workload results to be validated",
                 m_validate_workload);
    }

    auto get_stopping_no_except() -> std::optional<StoppingCriterion *> {
      if (m_stopping) {
        return m_stopping.get();
      }
      return {};
    }
    auto get_stopping() -> StoppingCriterion * {
      return m_stopping.get();
    }
    auto get_stats_engine() -> Stats::StatsEngine * {
      return m_stats_engine.get();
    }
    auto get_stats_engine_shared() -> std::shared_ptr<Stats::StatsEngine> {
      return m_stats_engine;
    }

    auto get_backend_options() -> OptionsMap {
      return m_backend_options;
    }

    auto get_batch_size() const -> size_t {
      return m_batch_size;
    }
    void set_batch_size(size_t newbatchsize) {
      m_batch_size = newbatchsize;
    }
    void print_callback(const RunReport &report) {
      if (m_printer) {
        m_printer->consume_single_run_report(report);
      }
    }
    auto get_warm_cool() const -> bool {
      return m_warm_cool;
    }
    auto get_warm_cool_timeout() const -> int {
      return m_warm_cool_timeout;
    }
    auto get_min_gpu_temp() const -> float {
      return m_min_gpu_temp;
    }
    auto get_max_gpu_temp() const -> float {
      return m_max_gpu_temp;
    }
    auto get_minimal_batch_duration() const -> float {
      return m_minimal_batch_duration;
    }
    auto get_dynamic_batch() const -> bool {
      return m_dynamic_batch;
    }
    auto get_max_blocking_queue() const -> int {
      return m_max_blocked_queue;
    }

    [[nodiscard]] virtual auto single_run(const std::optional<OptionsMap> &sweep_point) -> RunReport = 0;

  private:
    bool m_warm_cool = false;
    float m_min_gpu_temp = 45.0;
    float m_max_gpu_temp = 60.0;
    int m_warm_cool_timeout = 3;
    bool m_warmup = true;
    bool m_flush_l2 = true;
    int m_max_blocked_queue = 64;
    bool m_block = true;
    float m_block_duration_ms = DEFAULT_BLOCK_DURATION;
    bool m_first = true;
    bool m_dynamic_batch = false;
    bool m_validate_workload = false;
    float m_minimal_batch_duration = 10.0;
    size_t m_batch_size{25};
    OptionsMap stats_options;
    std::string m_name{DEFAULT_BENCHMARK_NAME};
    std::unique_ptr<StoppingCriterion> m_stopping;
    std::shared_ptr<Stats::StatsEngine> m_stats_engine;
    std::optional<SweepSpec> m_sweep_spec;
    OptionsMap m_backend_options;
    std::shared_ptr<IBenchmarkPrinter> m_printer;
  };

  template <typename BackendT>
  class Benchmark : public IBenchmark {
  public:
    using backend = BackendT;
    // Benchmark
    explicit Benchmark()
        : IBenchmark() {
      get_stats_engine()->template register_stat<Stats::ExecutionTimeVector>();
      get_stats_engine()->template register_stat<Stats::BatchSizeVector>();
      get_stats_engine()->template register_stat<Stats::BatchTimeVector>();
    };
    Benchmark(Benchmark &&) noexcept = default;
    auto operator=(Benchmark &&) noexcept -> Benchmark & = default;

    // Ensure copies are still deleted (good for safety)
    Benchmark(const Benchmark &) = delete;
    auto operator=(const Benchmark &) -> Benchmark & = delete;

    auto name() -> std::string override {
      if (m_workload) {
        return m_workload->algo() + m_workload->specialization() + get_m_name();
      }
      return get_m_name();
    }
    auto set_workload(std::shared_ptr<IWorkload<BackendT>> workload_impl) {
      m_workload = workload_impl;
    }
    auto run_benchmark() -> BenchmarkReport override {
      BenchmarkReport report;
      BackendT::instance()->apply_options(get_backend_options());
      report.hardware = this->get_hardware_info();
      auto sweeppoints = this->generate_sweep_points();
      for (const std::optional<OptionsMap> &sweep_point : sweeppoints) {
        if (ExecutionController::exit_requested()) {
          break;
        }
        report.results.push_back(this->single_run(sweep_point));
        print_callback(report.results.back());
      }
      return report;
    }
    auto get_workload_options() -> OptionsMap override {
      return m_workload->get_depedencies_options();
    };

  protected:
    auto generate_sweep_points() -> std::vector<OptionsMap> {
      if (!get_sweep_spec().has_value()) {
        return {{}};
      }
      SweepSpec spec = get_sweep_spec().value();
      auto resolved_axis = this->resolve_depedency_sweep_axis(spec.axes);
      return Sweep::get_sweep_points(spec.strategy, resolved_axis);
    }
    void register_options_dependencies() override {
      if (m_workload) {
        this->register_consumer(m_workload.get());
      }
      if (get_stopping_no_except().has_value()) {
        this->register_consumer(get_stopping());
      }
      this->register_consumer(BackendT::instance());
      this->register_consumer(get_stats_engine());
    }

    [[nodiscard]] auto single_run(const std::optional<OptionsMap> &sweep_point) -> RunReport override {
      const auto start_trial = std::chrono::steady_clock::now();
      this->apply_sweep_point(sweep_point);
      m_stream = BackendT::instance()->create_stream();
      std::shared_ptr<ITimer<BackendT>> timer = std::make_shared<Hardware::GpuTimer<BackendT>>();
      m_workload->set_timer(timer);
      check_components();
      int base_batch_size = get_batch_size();
      setup_metrics();
      get_stats_engine()->reset_engine();
      const auto tim0 = std::chrono::steady_clock::now();
      m_workload->setup_host();
      const auto tim1 = std::chrono::steady_clock::now();
      get_stats_engine()->template update_values<Stats::HostSetupTime>(
          std::chrono::duration<float, std::milli>(tim1 - tim0));

      auto setup_time = m_workload->timed_sync_setup_device(*m_stream);
      update_metrics();
      get_stats_engine()->template update_values<Stats::DeviceSetupTime>(setup_time);
      if (get_warmup()) {
        auto warmup_time = m_workload->timed_sync_run(*m_stream);
        get_stats_engine()->template update_values<Stats::WarmupTime>(warmup_time);
      }
      while (!get_stopping()->satisfied()) {
        if (ExecutionController::exit_requested()) {
          break;
        }
        if (get_warm_cool()) {
          Hardware::WarmingKernel<BackendT> warming_k;
          warming_k.alloc(*m_stream);
          auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(get_warm_cool_timeout());
          while (true) {
            if (ExecutionController::exit_requested()) {
              break;
            }
            int temp = get_stats_engine()->template get_result<Stats::DeviceTemperature<BackendT>>();
            if (std::chrono::steady_clock::now() > timeout) {
              warming_k.free();
              throw Errors::warm_cool_gpu_timeout(get_warm_cool_timeout());
            }
            if (temp < get_min_gpu_temp()) {
              for (int i = 0; i < 16; ++i) {
                warming_k.warm(*m_stream);
              }
              BackendT::synchronize(*m_stream);
            } else if (temp > get_max_gpu_temp()) {
              BackendT::instance()->cool_gpu(*m_stream);
            } else {
              break;
            }
          }
          warming_k.free();
        }
        if (get_block()) {
          m_blocker->block(*m_stream, get_block_duration());
        }
        m_workload->init_batch(*m_stream, get_batch_size(), get_block());
        for (int batch = 0; batch < get_batch_size(); batch++) {
          if (batch % get_max_blocking_queue() == 0 && get_block()) {
            m_blocker->unblock();
          }
          m_workload->reset_device(*m_stream);
          if (get_flush_l2()) {
            m_flusher->flush(*m_stream);
          }
          m_workload->timed_batch_run(*m_stream);
        }
        BackendT::get_last_error();
        if (get_block()) {
          m_blocker->unblock();
        }
        auto timer_v = m_workload->timed_run_elapsed_batch();
        for (int batch = 0; batch < get_batch_size(); batch++) {
          get_stats_engine()->template update_values<Stats::ExecutionTime>(timer_v[batch]);
          get_stats_engine()->compute_element_stats();
        }
        get_stats_engine()->template update_values<Stats::BatchSize>(get_batch_size());
        float batch_duration = sum(timer_v).count();
        get_stats_engine()->template update_values<Stats::BatchTime>(float_milliseconds(batch_duration));

        get_stats_engine()->compute_batch_stats();
        if (get_dynamic_batch()) {
          if (batch_duration < get_minimal_batch_duration()) {
            set_batch_size(get_batch_size() * 2);
          }
          if (batch_duration > get_minimal_batch_duration() * 2) {
            set_batch_size((get_batch_size() / 2) + 1);
          }
        }
      }
      post_all();
      auto fetch_time = m_workload->timed_sync_fetch_results(*m_stream);
      get_stats_engine()->template update_values<Stats::FetchResultsTime>(fetch_time);
      if (get_validate_workload()) {
        bool valid_run = m_workload->validate();
        if (!valid_run) {
          std::cout << "Warning, not able to validate Workload : " << m_workload->algo() + m_workload->specialization()
                    << '\n';
        }
      }
      m_workload->free();
      const auto end_trial = std::chrono::steady_clock::now();
      get_stats_engine()->template update_values<Stats::CpuTime>(
          std::chrono::duration<float, std::milli>(end_trial - start_trial));
      std::vector<Metric> metrics = {get_stats_engine()->get_metrics()};
      m_stream.reset();
      set_batch_size(base_batch_size);
      RunReport single_rep;
      single_rep.sweep_point = sweep_point;
      single_rep.measurements = metrics;
      return single_rep;
    }

  private:
    // Stats registry
    // Hardware specifics
    Hardware::L2Flusher<BackendT> *m_flusher = Hardware::L2Flusher<BackendT>::instance();
    Hardware::BlockingKernel<BackendT> *m_blocker = Hardware::BlockingKernel<BackendT>::instance();
    std::shared_ptr<typename BackendT::stream_t> m_stream;

    std::shared_ptr<IWorkload<BackendT>> m_workload;

    [[nodiscard]] auto get_hardware_info() const -> Hardware::HardwareInfo {
      return backend::instance()->get_device_info();
    };
    void apply_sweep_point(const std::optional<OptionsMap> &point) {
      if (point.has_value()) {
        this->apply_depedencies_options(point.value());
      }
    }
    virtual void update_metrics() {
      if (m_workload) {
        m_workload->update_metrics(get_stats_engine_shared());
      }
    }
    virtual void setup_metrics() {
      if (get_first()) {
        if (get_warmup()) {
          get_stats_engine()->template register_metric<Stats::WarmupTime>();
        }
        if (get_warm_cool()) {
          get_stats_engine()->template register_stat<Stats::DeviceTemperature<BackendT>>();
        }
        get_stats_engine()->template register_stat<Stats::Median>();

        get_stats_engine()->template register_metric<Stats::HostSetupTime>();
        get_stats_engine()->template register_metric<Stats::CpuTime>();
        get_stats_engine()->template register_metric<Stats::DeviceSetupTime>();
        get_stats_engine()->template register_metric<Stats::BatchSize>();
        get_stats_engine()->template register_metric<Stats::FetchResultsTime>();
        if (m_workload) {
          m_workload->setup_metrics(get_stats_engine_shared());
        }
        get_stats_engine()->set_options(get_stat_options());
        set_m_first(false);
      }
    }
    virtual void pre_all() {};
    virtual void post_all() {
      BackendT::synchronize(*m_stream);
    };
    void check_components() {
      if (!m_workload) {
        throw Errors::empty_workload_benchmark();
      }
      if (!get_stopping_no_except().has_value()) {
        throw Errors::empty_stopping_benchmark();
      }
    }
  };
} // namespace Baseliner

#endif // RUNNER_HPP