add_subdirectory(hardware)
add_subdirectory(stats)
target_sources(baseliner_core PRIVATE StoppingCriterion.cpp Error.cpp Options.cpp)