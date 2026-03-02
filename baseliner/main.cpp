
#include <argparse/argparse.hpp>
#include <baseliner/Handler.hpp>
#include <baseliner/RQ.hpp>
#include <baseliner/Recipe.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/Suite.hpp>
#include <baseliner/Version.hpp>
#include <baseliner/managers/Manager.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <utility>
#include <vector>
using namespace Baseliner;

__attribute__((weak)) auto main(int argc, char **argv) -> int { // NOLINT
  argparse::ArgumentParser program("Baseliner", Version::string(), argparse::default_arguments::all);
  argparse::ArgumentParser run_parser("run");
  run_parser.add_description("Runs the saved recipes");
  auto &run_group = run_parser.add_mutually_exclusive_group(false);
  run_group.add_argument("--config-files", "-cf")
      .nargs(argparse::nargs_pattern::at_least_one)
      .help("Running only the recipes inside one or multiple config files");
  run_group.add_argument("--replay-runs", "-rr")
      .nargs(argparse::nargs_pattern::at_least_one)
      .help("Replaying the reciped contained inside one or multiple results file");
  run_group.add_argument("--research-questions", "-rq")
      .nargs(argparse::nargs_pattern::at_least_one)
      .help("Running the research questions on given Cases");
  program.add_subparser(run_parser);

  argparse::ArgumentParser generate_parser("gen");
  generate_parser.add_description("Generate metadata or config files");
  auto &generate_group = generate_parser.add_mutually_exclusive_group(true);
  generate_group.add_argument("--metadata")
      .default_value("metadata.json")
      .nargs(1)
      .help("Generate the metadata file into the given file");
  generate_group.add_argument("--default-config-file", "--default-cf")
      .default_value("default-config.json")
      .nargs(1)
      .help("Generate the config files with all default values set");

  generate_group.add_argument("--saved-config-file", "--saved-cf")
      .default_value("saved-config.json")
      .nargs(1)
      .help("Generate the config file with the recipe saved in the binary");
  program.add_subparser(generate_parser);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << "\n";
    std::cerr << program;
    return 1;
  }

  Manager *manager = Manager::instance();
  if (program.is_subcommand_used("run")) {
    auto handler = Handler();
    if (run_parser.is_used("--config-files")) {
      auto config_files = run_parser.get<std::vector<std::string>>("--config-files");
      for (auto &config : config_files) {
        std::cout << "Runnning config : " << config << "...\n";
        Config parsed_config;
        file_to_config(parsed_config, config);
        Result result = handler.run_config(parsed_config);
        const std::string filename = "result-" + config + generate_uid() + ".json";
        result_to_file(result, filename);
        std::cout << "Result saved to " << filename << "\n";
      }
    } else if (run_parser.is_used("--replay-runs")) {
      auto replay_files = run_parser.get<std::vector<std::string>>("--replay-runs");
      for (auto &replay : replay_files) {
        Result parsed_result;
        std::cout << "Replaying result : " << replay << "...\n";
        file_to_result(parsed_result, replay);
        Result result = handler.replay_result(parsed_result);
        const std::string filename = "replay-" + replay + generate_uid() + ".json";
        result_to_file(result, filename);
        std::cout << "Result saved to " << filename << "\n";
      }
    } else if (run_parser.is_used("--research-questions")) {
      auto cases_names = run_parser.get<std::vector<std::string>>("--research-questions");
      manager->add_presets(get_rq_presets());
      for (auto &single_case : cases_names) {
        std::vector<Recipe> recipes;
        std::vector<std::string> backends = manager->get_backend_impl_for_case(single_case);
        if (backends.empty()) {
          std::cout << "No backend found for case :" << single_case;
          std::cout << "\nSkipping...\n";
        } else {
          for (const auto &backend : backends) {

            auto temp_recipe = get_rq_recipes(single_case, backend);
            recipes.insert(recipes.end(), temp_recipe.begin(), temp_recipe.end());
          }
          std::cout << "Runnning research questions for case " << single_case << "\n";
          Result result = handler.run_recipes(recipes);
          const std::string filename = "rq-" + single_case + generate_uid() + ".json";
          result_to_file(result, filename);
          std::cout << "Result saved to " << filename << "\n";
        }
      }
    } else {
      std::vector<Recipe> recipes = RecipeManager::get_recipes();
      size_t nb_recipes = recipes.size();
      if (nb_recipes > 0) {
        std::cout << "Running  " << recipes.size() << " saved Recipes" << "\n";
        auto result = handler.run_recipes(recipes);
        const std::string filename = "run-" + generate_uid() + ".json";
        result_to_file(result, filename);
        std::cout << "Result saved to " << filename << "\n";
      } else {
        std::cout << "No saved recipe...\n";
      }
    }
  } else if (program.is_subcommand_used("gen")) {
    if (generate_parser.is_used("--metadata")) {
      auto metadata_file = generate_parser.get<std::string>("--metadata");
      metadata_to_file(manager->generate_metadata(), metadata_file);
      std::cout << "Metadata file successfully saved to " << metadata_file << "\n";
    } else if (generate_parser.is_used("--default-config-file")) {
      auto config_file = generate_parser.get<std::string>("--default-config-file");
      config_to_file(manager->generate_default_config(), config_file);
      std::cout << "Default config file successfully saved to " << config_file << "\n";
    } else if (generate_parser.is_used("--saved-config-file")) {
      Config saved_config;
      saved_config.m_baseliner_version = Version::string();
      saved_config.m_presets = manager->get_default_presets();
      saved_config.m_recipes = RecipeManager::get_recipes();
      auto config_file = generate_parser.get<std::string>("--saved-config-file");
      config_to_file(saved_config, config_file);
      std::cout << "Saved config file successfully saved to " << config_file << "\n";
    }
  } else {
    std::cout << program << "\n";
  }
  /*
  Config config;
  file_to_config(config, "config.json");
  manager->add_presets(config.m_presets);
  RecipeManager::register_recipes(config.m_recipes);
  Handler handler;
  std::vector<Recipe> recipes = RecipeManager::get_recipes();
  std::cout << "[Baseliner] Total Registered Recipes: " << recipes.size() << "\n";

  Result result = handler.run_recipes(recipes);
  const std::string filename = generate_uid() + ".json";
  result_to_file(result, filename);
  */
  return 0;
};
