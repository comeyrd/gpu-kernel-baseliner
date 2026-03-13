#ifndef BASELINER_ERROR_HPP
#define BASELINER_ERROR_HPP
#include "baseliner/OptionTypes.hpp"
#include "baseliner/Serializer.hpp"
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace Baseliner {

  enum class ErrorCode {
    NotFound,
    BackendCaseBenchmarkNotFound,
    ComponentAlreadyExist,
    BenchmarkError,
    OptionsError,
    StoppingCriterionError,
    PresetError,
  };
  inline auto error_code_to_string(ErrorCode code) -> std::string {
    switch (code) {
    case ErrorCode::NotFound:
      return "NotFound";
    case ErrorCode::ComponentAlreadyExist:
      return "ComponentAlreadyExist";
    case ErrorCode::BenchmarkError:
      return "BenchmarkError";
    case ErrorCode::OptionsError:
      return "OptionsError";
    case ErrorCode::StoppingCriterionError:
      return "StoppingCriterionError";
    case ErrorCode::BackendCaseBenchmarkNotFound:
      return "BackendCaseBenchmarkNotFound";
    case ErrorCode::PresetError:
      return "PresetError";
    }
    return "Unknown";
  }

  class Error : public std::runtime_error {
  public:
    Error(ErrorCode code, const std::string &message)
        : std::runtime_error("[Baseliner][" + error_code_to_string(code) + "] " + message),
          m_code(code) {
    }

    [[nodiscard]] auto code() const -> ErrorCode {
      return m_code;
    }

    [[nodiscard]] auto with_context(const std::string &ctx) const -> Error {
      return {m_code, ctx + " -> " + strip_prefix()};
    }

  private:
    ErrorCode m_code;
    [[nodiscard]] auto strip_prefix() const -> std::string {
      return std::string(what()).substr(strlen("[Baseliner] "));
    }
  };

  namespace Errors {
    inline auto not_found(const std::string &kind, const std::string &name) -> Error {
      return {ErrorCode::NotFound, kind + " '" + name + "' not found"};
    }
    inline auto not_found_in(const std::string &kind, const std::string &name, const std::string &parent) -> Error {
      return {ErrorCode::NotFound, kind + " '" + name + "' not found in " + parent};
    }
    inline auto stat_not_found(const std::string &name) -> Error {
      return {ErrorCode::NotFound, "Stat '" + name + "' not found"};
    }
    inline auto stat_not_found_backend(const std::string &name, const std::string &backend) -> Error {
      return {ErrorCode::NotFound, "Stat '" + name + "' not found in Backend " + backend};
    }
    inline auto stat_not_found_either_general_or_backend(const std::string &name, const std::string &backend) -> Error {
      return {ErrorCode::NotFound, "Stat '" + name + "' not found neither in General Stats nor in Backend " + backend};
    }
    inline auto not_found_in_backend(const std::string &kind, const std::string &name, const std::string &backend)
        -> Error {
      return {ErrorCode::NotFound, kind + " '" + name + "' not found in Backend " + backend};
    }
    inline auto case_benchmark_not_found_in_backend(const std::string &kind, const std::string &name,
                                                    const std::string &backend) -> Error {
      return {ErrorCode::BackendCaseBenchmarkNotFound, kind + " '" + name + "' not found in Backend " + backend};
    }
    inline auto component_already_exists(const std::string &name, const std::string &kind) -> Error {
      return {ErrorCode::ComponentAlreadyExist,
              "The component " + name + " is already taken by a" + kind + "component"};
    }
    inline auto empty_case_benchmark() -> Error {
      return {ErrorCode::BenchmarkError, "Trying to run a benchmarking without a case set up"};
    }
    inline auto adding_option_outside_register_option() -> Error {
      return {ErrorCode::OptionsError, "add_options() was called outside register_options()"};
    }
    inline auto sweeping_error(const std::string &interface_name, const std::string &option_name) -> Error {
      return {ErrorCode::OptionsError, "Sweeping Error : Trying to sweep on option : " + interface_name + "." +
                                           option_name + " but no sweep policy was defined or given"};
    }
    inline auto empty_stat_engine_stopping() -> Error {
      return {ErrorCode::StoppingCriterionError,
              "The stopping criterion was called before the stats engine was provided"};
    }
    inline auto preset_not_subset_of(const OptionsMap &must_be_subset, const OptionsMap &original) -> Error {
      std::stringstream string_stream{};
      string_stream << "the given preset should be a subset of the object Option Schema \n";
      string_stream << "The given preset : \n";
      serialize(string_stream, must_be_subset);
      string_stream << "\n" << "The object preset \n";
      serialize(string_stream, original);
      return {ErrorCode::PresetError, string_stream.str()};
    }

  } // namespace Errors
} // namespace Baseliner

#endif // BASELINER_ERROR_HPP
