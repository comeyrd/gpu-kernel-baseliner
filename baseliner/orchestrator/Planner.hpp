#ifndef BASELINER_ORCHESTRATOR_PLANNER_HPP
#define BASELINER_ORCHESTRATOR_PLANNER_HPP
#include <baseliner/orchestrator/Plan.hpp>
#include <baseliner/orchestrator/Protocol.hpp>
#include <baseliner/registry/StorageManager.hpp>
namespace Baseliner::Planner {

  auto plan(const Protocol &protocol, const StorageManager *storage_manager) -> std::vector<Plan>;

} // namespace Baseliner::Planner

#endif // BASELINER_PLANNER_HPP