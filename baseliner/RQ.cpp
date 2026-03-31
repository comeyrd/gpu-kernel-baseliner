#include <baseliner/RQ.hpp>
#include <baseliner/core/Version.hpp>
#include <string>
#include <vector>

namespace Baseliner {
  namespace RQs {
    auto work_size(RQSize size) -> SweepHint {
      switch (size) {
      case RQSize::Small:
        return SweepHint{SweepPolicy::PowersOfTwo, "1", "128", "", {}};
      case RQSize::Medium:
        return SweepHint{SweepPolicy::PowersOfTwo, "1", "2048", "", {}};
      case RQSize::Large:
        return SweepHint{SweepPolicy::LinearRange, "1", "4096", "1000", {}};
      }
    }

    auto make_rq1(RQSize size) -> Recipe {
      SweepHint seed_hint;
      switch (size) {
      case RQSize::Small:
        seed_hint = SweepHint{SweepPolicy::LinearRange, "0", "10", "1", {}};
        break;
      case RQSize::Medium:
        seed_hint = SweepHint{SweepPolicy::LinearRange, "0", "100", "10", {}};
        break;
      case RQSize::Large:
        seed_hint = SweepHint{SweepPolicy::LinearRange, "0", "1000", "100", {}};
        break;
      }
      SweepHint work_size_hint = work_size(size);

      return Recipe{"Does different input values impact kernel execution times?",
                    {},
                    {},
                    {},
                    SweepSpec{SweepStrategy::FullGrid,
                              {SweepAxis{"Case", "seed", seed_hint}, SweepAxis{"Case", "work_size", work_size_hint}}}};
    }

    auto make_rq2(RQSize size) -> Recipe {
      SweepHint work_size_hint;
      switch (size) {
      case RQSize::Small:
        work_size_hint = SweepHint{SweepPolicy::PowersOfTwo, "1", "128", "", {}};
        break;
      case RQSize::Medium:
        work_size_hint = SweepHint{SweepPolicy::PowersOfTwo, "1", "2048", "", {}};
        break;
      case RQSize::Large:
        work_size_hint = SweepHint{SweepPolicy::LinearRange, "1", "4096", "250", {}};
        break;
      }
      return Recipe{"What impact has the work size on the kernel execution time?",
                    {},
                    {},
                    {},
                    SweepSpec{SweepStrategy::FullGrid, {SweepAxis{"Case", "work_size", work_size_hint}}}};
    }

    auto make_rq3(RQSize size) -> Recipe {
      SweepHint work_size_hint = work_size(size);

      return Recipe{"How does flushing the L2 cache impact kernel execution time?",
                    {},
                    {},
                    {},
                    SweepSpec{SweepStrategy::FullGrid,
                              {SweepAxis{"Benchmark", "flush", SweepHint{SweepPolicy::LinearRange, "0", "1", "1", {}}},
                               SweepAxis{"Case", "work_size", work_size_hint}}}};
    }

    auto make_rq4(RQSize size) -> Recipe {
      SweepHint work_size_hint = work_size(size);

      return Recipe{"What impact has enqueuing or not of kernels on execution time?",
                    {},
                    {},
                    {},
                    SweepSpec{SweepStrategy::FullGrid,
                              {SweepAxis{"Benchmark", "block", SweepHint{SweepPolicy::LinearRange, "0", "1", "1", {}}},
                               SweepAxis{"Case", "work_size", work_size_hint}}}};
    }

    auto make_rq5(RQSize size) -> Recipe {
      SweepHint work_size_hint = work_size(size);

      return Recipe{"How do warmups impact the kernel execution time?",
                    {},
                    {},
                    {},
                    SweepSpec{SweepStrategy::FullGrid,
                              {SweepAxis{"Benchmark", "warmup", SweepHint{SweepPolicy::LinearRange, "0", "1", "1", {}}},
                               SweepAxis{"Case", "work_size", work_size_hint}}}};
    }

  } // namespace RQs
  auto rq_recipes(RQSize size) -> std::unordered_map<std::string, Recipe> {
    return {
        {"RQ1", RQs::make_rq1(size)}, {"RQ2", RQs::make_rq2(size)}, {"RQ3", RQs::make_rq3(size)},
        {"RQ4", RQs::make_rq4(size)}, {"RQ5", RQs::make_rq5(size)},
    };
  }
  auto rq_protocol(RQSize size, std::vector<RecipeComponent> &cases, std::vector<RecipeComponent> &backends)
      -> Protocol {
    Protocol protocol;
    protocol.m_baseliner_version = Version::string();
    protocol.m_recipes = rq_recipes(size);
    Campaign base_campaign;
    base_campaign.m_backends = backends;
    base_campaign.m_cases = cases;
    base_campaign.m_on_incompatible = OnIncompatible::Skip;

    for (const auto &[name, _] : protocol.m_recipes) {
      Campaign temp = base_campaign;
      temp.m_name = name;
      temp.m_recipe = name;
      protocol.m_campaigns.push_back(temp);
    }
    return protocol;
  }

} // namespace Baseliner