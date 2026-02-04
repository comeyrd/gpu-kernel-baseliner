#ifndef JSON_HANDLER_HPP
#define JSON_HANDLER_HPP
#include <baseliner/Options.hpp>
#include <baseliner/research_questions/research_questions.hpp>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
NLOHMANN_JSON_NAMESPACE_BEGIN
template <>
struct adl_serializer<Baseliner::float_milliseconds> {
  static void to_json(json &j, const Baseliner::float_milliseconds &value) {
    j = value.count();
  }

  static void from_json(const json &j, Baseliner::float_milliseconds &value) {
    value = Baseliner::float_milliseconds(j.get<float>());
  }
};
NLOHMANN_JSON_NAMESPACE_END

namespace Baseliner {
  void to_json(json &j, const Option &opt);
  void from_json(const json &j, Option &opt);

  // namespace ResearchQuestions
  template <typename T>
  void save_to_json(std::ostream &os, T &obj) {
    json j;
    to_json(j, obj);
    os << std::setw(2) << j;
  }
} // namespace Baseliner

#endif // JSON_HANDLER_HPP