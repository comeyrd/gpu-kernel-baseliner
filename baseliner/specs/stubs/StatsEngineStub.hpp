#ifndef BASELINER_SPECS_STUBS_STATSENGINESTUB_HPP
#define BASELINER_SPECS_STUBS_STATSENGINESTUB_HPP
#include <any>
#include <baseliner/specs/stubs/StatsStubs.hpp>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <variant>
#include <vector>

namespace Baseliner::Stats {
  class StatsEngine {
  public:
    StatsEngine() = default;
    ~StatsEngine() = default;

    template <typename T>
    void register_metric(typename T::type initial_value = 0) {
      update_values<T>(initial_value);
    }

    template <typename T>
    void register_stat() {
    }

    template <typename T>
    void update_values(typename T::type value) {
      if constexpr (std::is_same_v<T, ByteNumbers>) {
        m_bytes = value;
      } else if constexpr (std::is_same_v<T, FLOPCount>) {
        m_flops = value;
      } else if constexpr (std::is_same_v<T, ExecutionTime>) {
        m_time_ms = value;
      }
    }

    void compute() {
      ArithmeticIntensity{}.calculate(m_intensity, m_bytes, m_flops);
      DataThroughput{}.calculate(m_data_bw, m_time_ms, m_bytes);
      FLOPTroughput{}.calculate(m_flop_bw, m_time_ms, m_flops);
    }

    template <typename T>
    [[nodiscard]] auto get() const -> typename T::type {
      if constexpr (std::is_same_v<T, ByteNumbers>)
        return m_bytes;
      else if constexpr (std::is_same_v<T, FLOPCount>)
        return m_flops;
      else if constexpr (std::is_same_v<T, ExecutionTime>)
        return m_time_ms;
      else if constexpr (std::is_same_v<T, ArithmeticIntensity>)
        return m_intensity;
      else if constexpr (std::is_same_v<T, DataThroughput>)
        return m_data_bw;
      else if constexpr (std::is_same_v<T, FLOPTroughput>)
        return m_flop_bw;
      else
        return typename T::type{};
    }

  private:
    size_t m_bytes = 0;
    size_t m_flops = 0;
    float m_time_ms = 0.0f;

    // Stockage des calculs
    float m_intensity = 0.0f;
    float m_data_bw = 0.0f;
    float m_flop_bw = 0.0f;
  };

} // namespace Baseliner::Stats
#endif // BASELINER_SPECS_STUBS_BACKENDSTUB_HPP
