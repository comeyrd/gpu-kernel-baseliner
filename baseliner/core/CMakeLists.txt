add_subdirectory(hardware)
add_subdirectory(stats)

target_sources(baseliner PRIVATE StoppingCriterion.cpp Error.cpp)
