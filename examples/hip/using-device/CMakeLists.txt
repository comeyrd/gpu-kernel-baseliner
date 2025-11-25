cmake_minimum_required(VERSION 3.15)
project(using-device LANGUAGES HIP)

include("../../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    find_package(baseliner REQUIRED)
endif()

set(sources main.hip)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

add_executable(using-device)
target_sources(using-device PRIVATE ${sources})
target_link_libraries(using-device PRIVATE baseliner::baseliner)