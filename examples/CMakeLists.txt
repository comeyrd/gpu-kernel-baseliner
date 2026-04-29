set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
cmake_minimum_required(VERSION 3.15)
project(examples LANGUAGES CXX)

include("../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    enable_language(CUDA)
    find_package(CUDAToolkit 12 REQUIRED)
    set(HAS_CUDA ON)
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    set(CMAKE_HIP_RUNTIME_LIBRARY Shared)
    set(HAS_HIP ON)
endif()

if(is_top_level)
find_package(baseliner REQUIRED)
endif()
add_subdirectory(ComputationWorkload)
add_subdirectory(MatMulWorkload)

if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp" "")
endif()

add_executable(baseliner-example "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp")
target_link_libraries(baseliner-example PRIVATE matmul computation)

