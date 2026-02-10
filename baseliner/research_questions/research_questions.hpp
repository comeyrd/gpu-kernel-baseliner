#ifndef BASELINER_RESEARCH_QUESTION_HPP
#define BASELINER_RESEARCH_QUESTION_HPP
#include <baseliner/Axe.hpp>
#include <string>
#include <vector>

namespace Baseliner::ResearchQuestions {
  struct Question {
    std::string m_name;
    std::string m_description;
    Axe m_axe;
  };
  const Question RQ1 = { // NOLINT
      "RQ1",
      "Does different input values impact kernel execution times ?",
      {"Kernel", "seed", {"123", "4444"}}};
  const Question RQ2 = { // NOLINT
      "RQ2",
      "What impact has the work size on the kernel execution time ?",
      {"Kernel", "work_size", {"1", "4", "16"}}};
  const Question RQ3 = { // NOLINT
      "RQ3",
      "How does flushing the L2 cache impact the kernel execution time",
      {"Runner", "flush", {"0", "1"}}};
  const Question RQ4 = { // NOLINT
      "RQ4",
      "What impact has the enqueing or not of kernels on it's execution time ?",
      {"Runner", "block", {"0", "1"}}};
  const Question RQ5 = { // NOLINT
      "RQ5",
      "How does warmups impact the kernel execution time ?",
      {"Runner", "warmup", {"0", "1"}}};

  // const std::vector<Question> AllRQs = {RQ1, RQ2, RQ3, RQ4, RQ5}; // NOLINT
} // namespace Baseliner::ResearchQuestions

#endif // BASELINER_RESEARCH_QUESTION_1_HPP