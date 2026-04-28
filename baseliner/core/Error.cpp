#include <baseliner/cli/Json.hpp>
#include <baseliner/core/Error.hpp>

namespace Baseliner::Errors {
  auto preset_not_subset_of(const OptionsMap &must_be_subset, const OptionsMap &original) -> Error {
    std::ostringstream string_stream{};
    string_stream << "the given preset should be a subset of the object Option Schema \n";
    string_stream << "The given preset : \n";
    Json::serialize(string_stream, must_be_subset);
    string_stream << "\n" << "The object preset \n";
    Json::serialize(string_stream, original);
    return {ErrorCode::PresetError, string_stream.str()};
  }
} // namespace Baseliner::Errors
