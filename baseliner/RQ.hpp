#ifndef BASELINER_RQ_HPP
#define BASELINER_RQ_HPP
#include <baseliner/Axe.hpp>
#include <baseliner/Benchmark.hpp>

namespace Baseliner {
  auto get_rq_presets() -> std::vector<PresetDefinition>;
  auto get_rq_recipes(const std::string &case_name, const std::string &backend_name) -> std::vector<Recipe>;
} // namespace Baseliner
#endif // BASELINER_RQ_HPP