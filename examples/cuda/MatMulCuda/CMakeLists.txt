cmake_minimum_required(VERSION 3.15)
project(matmul LANGUAGES CUDA)

include("../../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    find_package(baseliner REQUIRED)
endif()
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(sources MatMul.cu)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})
add_library(matmul SHARED)
target_sources(matmul PRIVATE ${sources})
target_link_libraries(matmul PRIVATE baseliner::baseliner)
target_include_directories(matmul PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})