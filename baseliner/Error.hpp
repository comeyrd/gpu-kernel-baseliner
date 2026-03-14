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
    AlreadyExists,
    BenchmarkError,
    OptionsError,
    StoppingCriterionError,
    PresetError,
    FileError,
    HardwareError,
    StatsEngineError,
  };
  inline auto error_code_to_string(ErrorCode code) -> std::string {
    switch (code) {
    case ErrorCode::NotFound:
      return "NotFound";
    case ErrorCode::AlreadyExists:
      return "AlreadyExists";
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
    case ErrorCode::FileError:
      return "FileError";
    case ErrorCode::HardwareError:
      return "HardwareError";
    case ErrorCode::StatsEngineError:
      return "StatsEngineError";
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
      return {ErrorCode::AlreadyExists, "The component " + name + " is already taken by a" + kind + "component"};
    }
    inline auto already_exist_in_backend(const std::string &kind, const std::string &name, const std::string &backend)
        -> Error {
      return {ErrorCode::AlreadyExists, kind + " '" + name + "' already exists in backend : " + backend};
    }
    inline auto already_exist(const std::string &kind, const std::string &name) -> Error {
      return {ErrorCode::AlreadyExists, kind + " '" + name + "' already "};
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
      std::ostringstream string_stream{};
      string_stream << "the given preset should be a subset of the object Option Schema \n";
      string_stream << "The given preset : \n";
      serialize(string_stream, must_be_subset);
      string_stream << "\n" << "The object preset \n";
      serialize(string_stream, original);
      return {ErrorCode::PresetError, string_stream.str()};
    }
    inline auto file_read_error(const std::string &filename) -> Error {
      return {ErrorCode::FileError, "Could not open file -" + filename + "- for reading"};
    }
    inline auto file_write_error(const std::string &filename) -> Error {
      return {ErrorCode::FileError, "Could not open file -" + filename + "- for writing"};
    }
    inline auto hardware_illegal_device_setting(int device, int device_count) -> Error {
      std::ostringstream string_stream{};
      string_stream << "Trying to set the device to " << device << " but there is only " << device_count
                    << " devices available";
      return {ErrorCode::HardwareError, string_stream.str()};
    }

    inline auto hardware_error_noexcept(const std::string &hardware, const std::string &error_code,
                                        const std::string &file, const std::string &line) -> std::string {
      std::ostringstream string_stream{};
      string_stream << hardware << " " << error_code << " in : " << file << " line " << line;
      return string_stream.str();
    }
    inline auto hardware_error(const std::string &hardware, const std::string &error_code, const std::string &file,
                               const std::string &line) -> Error {
      std::string error_str = hardware_error_noexcept(hardware, error_code, file, line);
      return {ErrorCode::HardwareError, error_str};
    }
    inline auto stat_dependencies_not_yet_computed(const std::string &inputs, const std::string &name) -> Error {
      std::ostringstream sstream;
      sstream << "Error in stat compute graph, one of inputs : ";
      sstream << inputs;
      sstream << " are not yet computed by needed by " << name;
      return {ErrorCode::StatsEngineError, sstream.str()};
    }
    inline auto circular_dependency_stat_engine() -> Error {
      return {ErrorCode::StatsEngineError, "Circular depedency detected in metrics graph"};
    }
    inline auto register_after_engine_built(const std::string &kind, const std::string &name) -> Error {
      return {ErrorCode::StatsEngineError, "Trying to register " + kind + " : " + name + " after the Engine is built"};
    }
    inline auto accessing_un_registered_thing(const std::string &kind, const std::string &name) -> Error {
      return {ErrorCode::StatsEngineError, "Trying to access " + kind + " : " + name + " but is not registered"};
    }

  } // namespace Errors
} // namespace Baseliner

#endif // BASELINER_ERROR_HPP
