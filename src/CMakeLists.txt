add_library(baseliner SHARED)
add_library(baseliner::baseliner ALIAS baseliner)
target_include_directories(baseliner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(backend)