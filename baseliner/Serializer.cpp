#include <baseliner/JsonHandler.hpp>
#include <baseliner/Serializer.hpp>
namespace Baseliner {

  template <typename T>
  void serialize(std::ostream &os, const T &obj) {
    json j;
    j = obj;
    os << std::setw(2) << j;
  }

  template void serialize<Metric>(std::ostream &os, const Metric &obj);
  template void serialize<Result>(std::ostream &os, const Result &obj);
  template void serialize<MetricData>(std::ostream &os, const MetricData &obj);
  template void serialize<MetricStats>(std::ostream &os, const MetricStats &obj);
  template void serialize<Option>(std::ostream &os, const Option &obj);

  template void serialize<std::vector<Result>>(std::ostream &os, const std::vector<Result> &obj);
  template void serialize<std::vector<Metric>>(std::ostream &os, const std::vector<Metric> &obj);

} // namespace Baseliner