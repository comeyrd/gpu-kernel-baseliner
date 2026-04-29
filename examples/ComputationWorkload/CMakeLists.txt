cmake_minimum_required(VERSION 3.15)
project(computation_kernel LANGUAGES CXX)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    find_package(baseliner REQUIRED)
endif()

add_library(computation SHARED)
target_link_libraries(computation PUBLIC baseliner::baseliner)
target_include_directories(computation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(BASELINER_HAS_CUDA)
  add_subdirectory(cuda)
endif()
if(BASELINER_HAS_HIP)
  add_subdirectory(hip)
endif()