cmake_minimum_required(VERSION 3.15)
project(simple_suite LANGUAGES CUDA)

include("../../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    find_package(baseliner REQUIRED)
endif()
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(sources simple_suite.cu)
include_directories(${CMAKE_INCLUDE_CURRENT_DIR})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
add_executable(simple_suite)
target_sources(simple_suite PRIVATE ${sources})
target_link_libraries(simple_suite PRIVATE baseliner::baseliner computation_kernel)
