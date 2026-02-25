#ifndef BASELINER_CASE_HPP
#define BASELINER_CASE_HPP
#include <baseliner/Options.hpp>

#include <baseliner/Timer.hpp>
#include <baseliner/backend/Backend.hpp>
#include <baseliner/stats/Stats.hpp>
#include <baseliner/stats/StatsEngine.hpp>
#include <memory>
#include <optional>
namespace Baseliner {
  constexpr int DEFAULT_SEED = 333;
  constexpr size_t DEFAULT_WORK_SIZE = 1;
  class IBaseCase : public IOption {
  public:
    void register_options() override {
      this->add_option("Case", "work_size", "The work size to apply, 1 = 32MFlop & 1 = 1MB", m_work_size);
      this->add_option("Case", "seed", "The seed used for the generation of input data", m_seed);
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

  private:
    size_t m_work_size = DEFAULT_WORK_SIZE;
    int m_seed = DEFAULT_SEED;
  };

  template <typename BackendT>
  class ICase : public Backend::GpuTimer<BackendT>, public IBaseCase {
  public:
    using backend = BackendT;
    using Backend::GpuTimer<BackendT>::time_elapsed;
    ICase() = default;
    ~ICase() override = default;
    virtual auto name() -> std::string = 0;
    virtual void setup(std::shared_ptr<typename BackendT::stream_t> stream) = 0;
    virtual void case_setup_metrics(std::shared_ptr<Stats::StatsEngine> &engine) {};
    virtual void case_update_metrics(std::shared_ptr<Stats::StatsEngine> &engine) {};
    virtual void reset_case(std::shared_ptr<typename BackendT::stream_t> stream) = 0;
    virtual void run_case(std::shared_ptr<typename BackendT::stream_t> stream) = 0;
    virtual void teardown(std::shared_ptr<typename BackendT::stream_t> stream) = 0;
    virtual auto validate_case() -> bool = 0;
    void setup_metrics(std::shared_ptr<Stats::StatsEngine> &engine) {
      std::optional<size_t> bytes = this->number_of_bytes();
      std::optional<size_t> flops = this->number_of_floating_point_operations();
      if (bytes.has_value()) {
        engine->register_metric<Stats::ByteNumbers>(bytes.value());
        m_bytes = true;
      }
      if (flops.has_value()) {
        engine->register_metric<Stats::FLOPCount>(flops.value());
        m_bytes = true;
      }
      this->case_setup_metrics(engine);
    };
    void update_metrics(std::shared_ptr<Stats::StatsEngine> &engine) override {
      std::optional<size_t> bytes = this->number_of_bytes();
      std::optional<size_t> flops = this->number_of_floating_point_operations();
      if (m_bytes) {
        engine->update_values<Stats::ByteNumbers>(bytes.value());
      }
      if (m_flops) {
        engine->update_values<Stats::FLOPCount>(flops.value());
      }
      this->case_update_metrics(engine);
    };
    virtual void timed_run(std::shared_ptr<typename BackendT::stream_t> stream) {
      this->measure_start(stream);
      run_case(stream);
      this->measure_stop(stream);
    };
    virtual void time_setup(std::shared_ptr<typename BackendT::stream_t> stream) {
      this->measure_start(stream);
      setup(stream);
      this->measure_stop(stream);
    };
    virtual void time_teardown(std::shared_ptr<typename BackendT::stream_t> stream) {
      this->measure_start(stream);
      teardown(stream);
      this->measure_stop(stream);
    };

  private:
    bool m_bytes = false;
    bool m_flops = false;
  };

} // namespace Baseliner
#endif // BASELINER_CASE_HPP