cmake_minimum_required(VERSION 3.15)
project(computation_kernel LANGUAGES CUDA)

include("../../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    find_package(baseliner REQUIRED)
endif()

set(sources ComputationKernel.cu)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})
add_library(computation_kernel SHARED)
target_sources(computation_kernel PRIVATE ${sources})
target_link_libraries(computation_kernel PRIVATE baseliner::baseliner)
target_include_directories(computation_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})