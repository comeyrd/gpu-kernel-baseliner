#include <string>
#ifndef BASELINER_SPECS_STUBS_STATSSTUB_HPP
#define BASELINER_SPECS_STUBS_STATSSTUB_HPP

namespace Baseliner::Stats {

  struct IBaseStat {
    virtual ~IBaseStat() = default;
    [[nodiscard]] virtual auto name() const -> std::string = 0;
    [[nodiscard]] virtual auto unit() const -> std::string = 0;
  };

  template <typename Derived, typename T>
  struct Imetric : public IBaseStat {
    using type = T;
  };

  template <typename Derived, typename T, typename... Deps>
  struct IStat : public IBaseStat {
    using type = T;
    virtual void calculate(T &value_to_update, const typename Deps::type &...deps) = 0;
  };

  class ByteNumbers : public Imetric<ByteNumbers, size_t> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "memory_usage";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "bytes";
    }
  };

  class FLOPCount : public Imetric<FLOPCount, size_t> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "arithmetic_usage";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "Flops";
    }
  };

  class ExecutionTime : public Imetric<ExecutionTime, float> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "execution_time";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
  };

  class ArithmeticIntensity : public IStat<ArithmeticIntensity, float, ByteNumbers, FLOPCount> {
  public:
    void calculate(float &val, const size_t &bytes, const size_t &flops) override {
      val = (bytes > 0) ? static_cast<float>(flops) / bytes : 0.0f;
    }
    [[nodiscard]] auto name() const -> std::string override {
      return "arithmetic_intensity";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "Flops/Byte";
    }
  };

  class DataThroughput : public IStat<DataThroughput, float, ExecutionTime, ByteNumbers> {
  public:
    void calculate(float &val, const float &ms, const size_t &bytes) override {
      val = (ms > 0) ? (static_cast<float>(bytes) / 1e6f) / ms : 0.0f;
    }
    [[nodiscard]] auto name() const -> std::string override {
      return "memory_bandwidth";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "GB/s";
    }
  };
  class FLOPThroughput : public IStat<FLOPThroughput, float, ExecutionTime, FLOPCount> {
  public:
    void calculate(FLOPThroughput::type &value_to_update, const float &mean,
                   const typename FLOPCount::type &nb_flops) override {
      auto flops = static_cast<double>(nb_flops);
      auto miliseconds = static_cast<double>(mean);
      if (miliseconds > 0) {
        value_to_update = static_cast<float>(flops / (miliseconds * 1e6));
      } else {
        value_to_update = 0.0F;
      }
    }
    [[nodiscard]] auto name() const -> std::string override {
      return "arithmetic_bandwidth";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "GFLOP/S";
    }
  };

} // namespace Baseliner::Stats
#endif // BASELINER_SPECS_STUBS_STATSSTUB_HPP
