cmake_minimum_required(VERSION 3.15)
project(using_harness LANGUAGES HIP)

include("../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
find_package(baseliner REQUIRED)
endif()
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
add_subdirectory(ComputationKernel)
add_subdirectory(MatMulHip)

if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp" "")
endif()

add_executable(baseliner-example "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp")
target_link_libraries(baseliner-example PRIVATE baseliner::baseliner matmul computation_kernel)
