#ifndef BASELINER_CORE_AXESWEEPING_HPP
#define BASELINER_CORE_AXESWEEPING_HPP
#include <baseliner/core/OptionTypes.hpp>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace Baseliner {
  enum class SweepStrategy : char {
    FullGrid
  };
  DESCRIBE_ENUM(SweepStrategy, ENUM_VALUE(FullGrid))

  enum class SweepPolicy : char {
    PowersOfTwo,
    LinearRange,
    Enumerated
  };
  DESCRIBE_ENUM(SweepPolicy, ENUM_VALUE(PowersOfTwo), ENUM_VALUE(LinearRange), ENUM_VALUE(Enumerated))

  struct SweepHint {
    SweepPolicy m_policy;
    std::string m_min;
    std::string m_max;
    std::string m_step;
    std::vector<std::string> m_enumerated;
  };
  DESCRIBE(SweepHint, FIELD(m_policy), FIELD(m_min), FIELD(m_max), FIELD(m_step), FIELD(m_enumerated))

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
    std::optional<SweepHint> m_hint;
  };
  DESCRIBE(SweepAxis, FIELD(m_interface), FIELD(m_option), FIELD(m_hint))

  struct ResolvedAxis {
    std::string m_interface;
    std::string m_option;
    std::vector<std::string> value;
  };
  DESCRIBE(ResolvedAxis, FIELD(m_interface), FIELD(m_option), FIELD(value))

  struct SweepSpec {
    SweepStrategy m_strategy;
    std::vector<SweepAxis> m_axes;
  };
  DESCRIBE(SweepSpec, FIELD(m_strategy), FIELD(m_axes))

  using SweepHintMap = std::unordered_map<std::string, std::unordered_map<std::string, SweepHint>>;

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

      template <>
      struct Sweeper<bool> {
        static auto generate(const TypedSweepHint<bool> &hint) -> std::vector<bool> {
          if (hint.m_policy == SweepPolicy::Enumerated) {
            return hint.m_enumerated;
          }
          if (hint.m_min == hint.m_max) {
            return {hint.m_min};
          }
          return {false, true};
        }
      };
    } // namespace Detail

    template <typename T>
    auto generate_sweep_values(const TypedSweepHint<T> &typed) -> std::vector<T> {
      return Detail::Sweeper<T>::generate(typed);
    }

    inline auto get_sweep_points(const SweepStrategy &strategy, const std::vector<ResolvedAxis> &axes)
        -> std::vector<OptionsMap> {

      if (axes.empty()) {
        return {};
      }

      std::vector<OptionsMap> result = {{}}; // Start with one empty map

      switch (strategy) {
      case SweepStrategy::FullGrid: {
        for (const ResolvedAxis &axis : axes) {
          std::vector<OptionsMap> next;

          for (const OptionsMap &existing : result) {
            for (const std::string &val : axis.value) {
              OptionsMap entry = existing; // copy current combination
              entry[axis.m_interface][axis.m_option] = Option{"", val};
              next.push_back(std::move(entry));
            }
          }

          result = std::move(next);
        }
        break;
      }
      }

      return result;
    }
  } // namespace Sweep
} // namespace Baseliner

#endif