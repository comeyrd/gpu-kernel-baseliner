#include "ComputationKernel.hpp"
#include <baseliner/Options.hpp>
#include <baseliner/Runner.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <iostream>

#include <string>
#include <vector>
inline std::ostream &operator<<(std::ostream &os, const Baseliner::OptionsMap &option_map) {
  os << "{" << std::endl;
  for (const auto &[key, val] : option_map) {
    os << "  " << key << " : {" << std::endl;
    for (auto [name, opt] : val) {
      os << "    " << name << " | " << opt.m_description << " | " << opt.m_value << " ," << std::endl;
    }
    os << "  }," << std::endl;
  }
  os << "}" << std::endl;
  return os;
}

void help(Baseliner::OptionsMap &options_map) {
  std::cout << "Usage of simple_cli : " << std::endl;
  for (auto &[interface_name, interface] : options_map) {
    std::cout << "  --" << interface_name << std::endl;
    for (auto &[option_name, option] : interface) {
      std::cout << "    " << std::string(interface_name.length(), ' ') << "." << option_name << " | "
                << option.m_description << " | default : " << option.m_value << std::endl;
    }
  }
}
void usage(std::vector<std::string> args, Baseliner::OptionsMap &options_map) {
  std::cout << "Bad usage of simple_cli : ";
  for (auto arg : args)
    std::cout << arg << " ";
  std::cout << std::endl;
  help(options_map);
}

int main(int argc, char **argv) {
  std::cout << "MiniCli" << std::endl;
  auto runner_act = Baseliner::Runner<ComputationKernel, Baseliner::Backend::CudaBackend>();
  Baseliner::OptionsMap omap = runner_act.gather_options();

  Baseliner::OptionsMap user_options;
  std::vector<std::string> args(argv + 1, argv + argc);
  std::string current_arg;
  bool current_bool = false;
  for (auto &i : args) {
    if (i == "-h" || i == "--help") {
      std::cout << omap << std::endl;
      help(omap);
      exit(0);
    }
    if (!current_bool) {
      if (i.substr(0, 2) == "--") {
        current_arg = i.substr(2);
        current_bool = true;
        continue;
      } else {
        usage(args, omap);
        exit(1);
      }
    } else {
      size_t pos = current_arg.find('.');
      std::string pre = current_arg.substr(0, pos);
      std::string post = current_arg.substr(pos + 1);
      if (auto option_iterator = omap.find(pre); option_iterator != omap.end()) {
        auto [name, option] = *option_iterator;
        if (auto inner_iter = option.find(post); inner_iter != option.end()) {
          auto [inner_name, inner_opt] = *inner_iter;
          user_options[name][inner_name] = Baseliner::Option{inner_opt.m_description, i};
          current_bool = false;
          continue;
        }
        usage(args, omap);
        exit(1);
      } else {
        usage(args, omap);
        exit(1);
      }
    }
  }
  std::cout << user_options;
  runner_act.propagate_options(user_options);
  serialize(std::cout, runner_act.run());
  std::cout << std::endl;
  /*
  std::vector<float_milliseconds> res = runner_act.run();
  std::cout << res << std::endl;
  */
}
