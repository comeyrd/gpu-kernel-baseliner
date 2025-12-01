cmake_minimum_required(VERSION 3.15)
project(using_kernel LANGUAGES CUDA)

include("../../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    find_package(baseliner REQUIRED)
endif()

set(sources main.cu)
include_directories(${CMAKE_INCLUDE_CURRENT_DIR}../)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
add_executable(using_kernel)
target_sources(using_kernel PRIVATE ${sources})
target_link_libraries(using_kernel PRIVATE baseliner::baseliner computation_kernel)
