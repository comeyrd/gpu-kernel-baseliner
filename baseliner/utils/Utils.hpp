#ifndef BASELINER_UTILS_UTILS_HPP
#define BASELINER_UTILS_UTILS_HPP
#include <string>
namespace Baseliner::Utils {

  auto gen_uuid() -> std::string;
  auto get_datetime() -> std::string;
} // namespace Baseliner::Utils
#endif // BASELINER_UTILS_UTILS_HPP