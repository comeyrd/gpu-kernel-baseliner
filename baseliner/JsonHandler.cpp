#include "JsonHandler.hpp"
namespace Baseliner {
  void to_json(json &j, const Option &opt) {
    // j = json{{"description", opt.m_description}, {"value", opt.m_value}}; //We don't need the description in the JSON
    j = opt.m_value;
  }
  void from_json(const json &j, Option &opt) {
    // j.at("description").get_to(opt.m_description);// Same as above
    j.get_to(opt.m_value);
  }
} // namespace Baseliner