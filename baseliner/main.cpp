
#include <argparse/argparse.hpp>
#include <baseliner/Orchestrator.hpp>
#include <baseliner/Output.hpp>
#include <baseliner/Protocol.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/State.hpp>
#include <baseliner/Version.hpp>
#include <csignal>
#include <iostream>
#include <vector>
using namespace Baseliner;

__attribute__((weak)) int main(int argc, char **argv) { // NOLINT
  std::signal(SIGINT, ExecutionControllerSignalHandler);
  argparse::ArgumentParser program("Baseliner", Version::string(), argparse::default_arguments::all);
  argparse::ArgumentParser run_parser("run");
  run_parser.add_description("Runs the saved recipes");
  auto &run_group = run_parser.add_mutually_exclusive_group(false);
  run_group.add_argument("--protocol-files", "-pf")
      .nargs(argparse::nargs_pattern::at_least_one)
      .help("Running only the recipes inside one or multiple protocol files");
  run_group.add_argument("--replay-runs", "-rr")
      .nargs(argparse::nargs_pattern::at_least_one)
      .help("Replaying the reciped contained inside one or multiple results file");
  run_group.add_argument("--research-questions", "-rq")
      .nargs(argparse::nargs_pattern::at_least_one)
      .help("Running the research questions on given Cases");
  run_parser.add_argument("--load-preset-from-protocol-file", "--load-preset")
      .help("Load presets into baseliner with a protocol file (will ignore any recipes defined inside)")
      .nargs(1);
  program.add_subparser(run_parser);

  argparse::ArgumentParser generate_parser("gen");
  generate_parser.add_description("Generate metadata or protocol files");
  auto &generate_group = generate_parser.add_mutually_exclusive_group(true);
  generate_group.add_argument("--metadata")
      .default_value("metadata.json")
      .nargs(1)
      .help("Generate the metadata file into the given file");
  generate_group.add_argument("--default-protocol-file", "--default-pf")
      .default_value("default-protocol.json")
      .nargs(1)
      .help("Generate the protocol files with all default values set");

  generate_group.add_argument("--saved-protocol-file", "--saved-pf")
      .default_value("saved-protocol.json")
      .nargs(1)
      .help("Generate the protocol file with the recipe saved in the binary");
  program.add_subparser(generate_parser);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << "\n";
    std::cerr << program;
    return 1;
  }

  if (program.is_subcommand_used("run")) {
    if (run_parser.is_used("--load-preset")) {
      throw Errors::not_implemented("--load-preset not implemented yet");
      auto preset_pf = run_parser.get<std::string>("--load-preset");
      auto preset_protocol = from_file<Protocol>(preset_pf);
      Orchestrator::load_presets(preset_protocol);
    }
    if (run_parser.is_used("--protocol-files")) {
      auto protocol_files = run_parser.get<std::vector<std::string>>("--protocol-files");
      for (auto &protocol : protocol_files) {
        if (ExecutionController::exit_requested()) {
          break;
        }
        std::cout << "Runnning protocol : " << protocol << "...\n";
        auto parsed_protocol = from_file<Protocol>(protocol);
        Report report = Orchestrator::run_protocol(parsed_protocol);
        // TODO fix generate_uid
        //  const std::string filename = "result-" + generate_uid() + ".json";
        const std::string filename = "result.json";
        to_file(report, filename);
        std::cout << "Report saved to " << filename << "\n";
      }
    } else if (run_parser.is_used("--replay-runs")) {
      auto replay_files = run_parser.get<std::vector<std::string>>("--replay-runs");
      for (auto &replay : replay_files) {
        if (ExecutionController::exit_requested()) {
          break;
        }
        auto parsed_report = from_file<Report>(replay);
        Report report = Orchestrator::replay_runs(parsed_report);
        const std::string filename = "result.json";
        to_file(report, filename);
        std::cout << "Report saved to " << filename << "\n";
      }
    } else if (run_parser.is_used("--research-questions")) {
      auto cases_names = run_parser.get<std::vector<std::string>>("--research-questions");
      Report report = Orchestrator::run_research_questions(cases_names);
      const std::string filename = "result.json";
      to_file(report, filename);
      std::cout << "Report saved to " << filename << "\n";
    }
  }

  else if (program.is_subcommand_used("gen")) {
    if (generate_parser.is_used("--metadata")) {
      throw Errors::not_implemented("Metadata generation is not implemented");
    } else if (generate_parser.is_used("--default-protocol-file")) {
      Protocol protocol = Orchestrator::get_default_protocol();
      to_file(protocol, "default-protocol.json");
    }
    /*else if (generate_parser.is_used("--saved-protocol-file")) {
      Protocol saved_protocol;
      saved_protocol.m_baseliner_version = Version::string();
      saved_protocol.m_presets = manager->get_all_preset_definitions();
      saved_protocol.m_recipes = RecipeManager::get_recipes();
      auto protocol_file = generate_parser.get<std::string>("--saved-protocol-file");
      protocol_to_file(saved_protocol, protocol_file);
      std::cout << "Saved protocol file successfully saved to " << protocol_file << "\n";
    }*/
    else {
      std::cout << program << "\n";
    }
  } else {
    std::cout << program << "\n";
  }
  return 0;
};
