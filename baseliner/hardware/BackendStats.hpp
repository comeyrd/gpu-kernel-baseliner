#ifndef BASELINER_BACKEND_STATS_HPP
#define BASELINER_BACKEND_STATS_HPP
#include <baseliner/stats/Stats.hpp>

namespace Baseliner::Stats {

  template <typename BackendT, typename OutputTag, typename ValueType, typename... InputTags>
  class BackendStat : public IStat<OutputTag, ValueType, InputTags...> {
  public:
    using backend = BackendT;
  };

  template <typename BackendT>
  class ClockFrequency : public BackendStat<BackendT, ClockFrequency<BackendT>, float> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "clock_frenquency";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "GHz";
    }
    void calculate(typename ClockFrequency<BackendT>::type &value_to_update) override;
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::EVERY_TICK;
    };
  };

  template <typename BackendT>
  class ClockFrequencyVector
      : public BackendStat<BackendT, ClockFrequencyVector<BackendT>, std::vector<float>, ClockFrequency<BackendT>> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "clock_frequency_vector";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "GHz";
    }
    void calculate(typename ClockFrequencyVector<BackendT>::type &value_to_update,
                   typename ClockFrequency<BackendT>::type const &input) override {
      value_to_update.push_back(input);
    };
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::EVERY_TICK;
    };
  };

  template <typename BackendT>
  class DeviceTemperature : public BackendStat<BackendT, DeviceTemperature<BackendT>, int> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "temperature";
    };
    [[nodiscard]] auto unit() const -> std::string override {
      return "°C";
    }
    void calculate(typename DeviceTemperature<BackendT>::type &value_to_update) override;
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::EVERY_TICK;
    };
  };
  template <typename BackendT>
  class DeviceTemperatureVector
      : public BackendStat<BackendT, DeviceTemperatureVector<BackendT>, std::vector<int>, DeviceTemperature<BackendT>> {
    [[nodiscard]] auto name() const -> std::string override {
      return "temperature_vector";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "°C";
    }
    void calculate(typename DeviceTemperatureVector<BackendT>::type &value_to_update,
                   const typename DeviceTemperature<BackendT>::type &temp) override {
      value_to_update.push_back(temp);
    };
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::EVERY_TICK;
    };
  };
  template <typename BackendT>
  class DevicePowerUtilization : public BackendStat<BackendT, DevicePowerUtilization<BackendT>, float> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "power_utilization";
    };
    [[nodiscard]] auto unit() const -> std::string override {
      return "%";
    }
    void calculate(typename DevicePowerUtilization<BackendT>::type &value_to_update) override;
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::EVERY_TICK;
    };
  };
  template <typename BackendT>
  class DevicePowerUtilizationVector : public BackendStat<BackendT, DevicePowerUtilizationVector<BackendT>,
                                                          std::vector<float>, DevicePowerUtilization<BackendT>> {
    [[nodiscard]] auto name() const -> std::string override {
      return "power_utilization_vector";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "%";
    }
    void calculate(typename DevicePowerUtilizationVector<BackendT>::type &value_to_update,
                   const typename DevicePowerUtilization<BackendT>::type &temp) override {
      value_to_update.push_back(temp);
    };
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::EVERY_TICK;
    };
  };

} // namespace Baseliner::Stats
#endif // BASELINER_BACKEND_STATS_HPP