cmake_minimum_required(VERSION 3.15)
project(using_device LANGUAGES HIP)

include("../../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    find_package(baseliner REQUIRED)
endif()

set(sources main.hip)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

add_executable(using_device)
target_sources(using_device PRIVATE ${sources})
target_link_libraries(using_device PRIVATE baseliner::baseliner)
