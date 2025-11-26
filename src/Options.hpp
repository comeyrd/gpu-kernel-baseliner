#ifndef OPTIONS_HPP
#define OPTIONS_HPP
#include <string>
#include <unordered_map>
#include <vector>
namespace Baseliner {

  class Option {
    public:
    std::string m_name;
    std::string m_description;
    std::string m_value;
  };

  using InterfaceOptions = std::vector<Option>;
  using OptionsMap = std::unordered_map<std::string, InterfaceOptions>;

  class OptionConsumer {
  public:
    virtual std::pair<std::string,InterfaceOptions> describe_options() = 0;
    virtual void apply_options(InterfaceOptions &options) = 0;
  };

} // namespace Baseliner
#endif // OPTIONS_HPP