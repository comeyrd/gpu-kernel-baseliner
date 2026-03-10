#ifndef BASELINER_AXE_SWEEPING_HPP
#define BASELINER_AXE_SWEEPING_HPP

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace Baseliner {
  enum class SweepStrategy : char {
    Grid,
    Paired
  };

  enum class SweepPolicy : char {
    PowersOfTwo,
    LinearRange,
    Enumerated
  };

  struct SweepHint {
    SweepPolicy m_policy;
    std::string m_min;
    std::string m_max;
    std::string m_step;
    std::vector<std::string> m_enumerated;
  };

  template <typename T>
  struct TypedSweepHint {
    SweepPolicy m_policy;
    T m_min;
    T m_max;
    T m_step;
    std::vector<T> m_enumerated;
  };

  struct SweepAxis {
    std::string m_interface;
    std::string m_option;
    SweepHint m_hint;
  };
  struct SweepSpec {
    SweepStrategy m_strategy;
    std::vector<SweepAxis> m_axes;
  };

  namespace Sweep {
    namespace Detail {

      template <typename T>
      struct Sweeper {
        static auto generate(const TypedSweepHint<T> &hint) -> std::vector<T> {
          if (hint.m_policy == SweepPolicy::Enumerated) {
            return hint.m_enumerated;
          }

          std::vector<T> result;
          if (hint.m_policy == SweepPolicy::LinearRange) {
            if (hint.m_step <= static_cast<T>(0)) {
              throw std::invalid_argument("Step must be > 0");
            }
            for (T value = hint.m_min; value <= hint.m_max; value += hint.m_step) {
              result.push_back(value);
            }
          } else if (hint.m_policy == SweepPolicy::PowersOfTwo) {
            if (hint.m_min <= static_cast<T>(0)) {
              throw std::invalid_argument("Min must be > 0");
            }
            for (T value = hint.m_min; value <= hint.m_max; value *= static_cast<T>(2)) {
              result.push_back(value);
            }
          }
          return result;
        }
      };
    } // namespace Detail

    template <typename T>
    auto generate_sweep_values(const TypedSweepHint<T> &typed) -> std::vector<T> {
      return Detail::Sweeper<T>::generate(typed);
    }
  } // namespace Sweep
} // namespace Baseliner

#endif