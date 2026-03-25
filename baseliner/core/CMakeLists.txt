add_subdirectory(hardware)
add_subdirectory(stats)

target_sources(baseliner PRIVATE Options.cpp StoppingCriterion.cpp)
