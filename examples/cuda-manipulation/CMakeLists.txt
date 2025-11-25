cmake_minimum_required(VERSION 3.15)
project(cuda-manipulation LANGUAGES CUDA)

include("../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    find_package(baseliner REQUIRED)
endif()

set(sources main.cu)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

add_executable(cuda-manipulation)
target_sources(cuda-manipulation PRIVATE ${sources})
target_link_libraries(cuda-manipulation PRIVATE baseliner::baseliner)