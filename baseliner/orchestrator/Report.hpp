#ifndef BASELINER_ORCHESTRATOR_REPORT_HPP
#define BASELINER_ORCHESTRATOR_REPORT_HPP
#include <baseliner/orchestrator/Plan.hpp>
#include <string>
#include <vector>
namespace Baseliner {
  struct RunReport {
    Plan m_plan;
    BenchmarkReport m_benchmark_report;
  };
  struct Report {
    std::string m_baseliner_version;
    std::string m_git_version;
    std::string m_datetime;
    std::vector<RunReport> m_runs;
  };
} // namespace Baseliner

#endif