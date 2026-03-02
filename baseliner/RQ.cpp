#include <baseliner/Axe.hpp>
#include <baseliner/ConfigFile.hpp>
#include <baseliner/Recipe.hpp>
#include <string>
#include <vector>

namespace Baseliner {
  namespace ResearchQuestions {

    const auto rq1 = []() -> PresetDefinition {
      SingleAxe axe;
      axe.set_interface_name("Case");
      axe.set_option_name("seed");
      axe.set_values({"123", "4444", "2133"});
      PresetDefinition preset;
      preset.m_preset_name = "RQ1";
      preset.m_implementation_name = "SingleAxeSuite";
      preset.m_description = "Does different input values impact kernel execution times ?";
      preset.m_options = axe.gather_options();
      return preset;
    };
    const auto rq2 = []() -> PresetDefinition {
      SingleAxe axe;
      axe.set_interface_name("Case");
      axe.set_option_name("work_size");
      axe.set_values({"1", "2", "4", "8", "16", "32", "64", "128", "256", "512"});
      PresetDefinition preset;
      preset.m_preset_name = "RQ2";
      preset.m_implementation_name = "SingleAxeSuite";
      preset.m_description = "What impact has the work size on the kernel execution time ?";
      preset.m_options = axe.gather_options();
      return preset;
    };
    const auto rq3 = []() -> PresetDefinition {
      SingleAxe axe;
      axe.set_interface_name("Benchmark");
      axe.set_option_name("flush");
      axe.set_values({"0", "1"});
      PresetDefinition preset;
      preset.m_preset_name = "RQ3";
      preset.m_implementation_name = "SingleAxeSuite";
      preset.m_description = "How does flushing the L2 cache impact the kernel execution time";
      preset.m_options = axe.gather_options();
      return preset;
    };
    const auto rq4 = []() -> PresetDefinition {
      SingleAxe axe;
      axe.set_interface_name("Benchmark");
      axe.set_option_name("block");
      axe.set_values({"0", "1"});
      axe.gather_options();
      PresetDefinition preset;
      preset.m_preset_name = "RQ4";
      preset.m_implementation_name = "SingleAxeSuite";
      preset.m_description = "What impact has the enqueing or not of kernels on it's execution time ?";
      preset.m_options = axe.gather_options();
      return preset;
    };
    const auto rq5 = []() -> PresetDefinition {
      SingleAxe axe;
      axe.set_interface_name("Benchmark");
      axe.set_option_name("warmup");
      axe.set_values({"0", "1"});
      PresetDefinition preset;
      preset.m_preset_name = "RQ5";
      preset.m_implementation_name = "SingleAxeSuite";
      preset.m_description = "How does warmups impact the kernel execution time ?";
      preset.m_options = axe.gather_options();
      return preset;
    };
    const static PresetDefinition all_rq[5] = {rq1(), rq2(), rq3(), rq4(), rq5()};
    const static std::string_view all_rq_names[5] = {"RQ1", "RQ2", "RQ3", "RQ4", "RQ5"};
  } // namespace ResearchQuestions
  auto get_rq_presets() -> std::vector<PresetDefinition> {
    std::vector<PresetDefinition> presets;
    for (const auto &preset : ResearchQuestions::all_rq) {
      presets.push_back(preset);
    }
    return presets;
  }
  auto get_rq_recipes(const std::string &case_name, const std::string &backend_name) -> std::vector<Recipe> {
    std::vector<Recipe> recipes;
    Recipe baserecipe;
    baserecipe.m_backend = w_default_preset(backend_name);
    baserecipe.m_case = w_default_preset(case_name);
    baserecipe.m_stopping = w_default_preset("StoppingCriterion");
    baserecipe.m_benchmak = w_default_preset("Benchmark");
    baserecipe.m_stats = w_default_preset("Stat");
    for (const std::string_view &research_q : ResearchQuestions::all_rq_names) {
      Recipe temp_recipe = baserecipe;
      temp_recipe.m_suite = WithPreset{"SingleAxeSuite", std::string(research_q)};
      recipes.push_back(temp_recipe);
    }
    return recipes;
  }
} // namespace Baseliner
