cmake_minimum_required(VERSION 3.15)
project(baseliner-device-manipulation LANGUAGES CXX)

include("../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    find_package(baseliner REQUIRED)
endif()

set(sources main.cpp)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

add_executable(baseliner-device-manipulation)
target_sources(baseliner-device-manipulation PRIVATE ${sources})
target_link_libraries(baseliner-device-manipulation PRIVATE baseliner::baseliner)