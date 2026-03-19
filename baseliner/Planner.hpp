#ifndef BASELINER_PLANNER_HPP
#define BASELINER_PLANNER_HPP
#include <baseliner/Protocol.hpp>
#include <baseliner/managers/StorageManager.hpp>
namespace Baseliner::Planner {

  auto plan(const Protocol &protocol, const StorageManager *storage_manager) -> std::vector<Plan>;

} // namespace Baseliner::Planner

#endif // BASELINER_PLANNER_HPP