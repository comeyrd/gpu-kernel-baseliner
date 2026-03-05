
cmake_minimum_required(VERSION 3.15)
project(nothing-kernel VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FOUND_DEVICE_COMPILER OFF)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    enable_language(CUDA)
    set(HAS_CUDA ON)
    set(FOUND_DEVICE_COMPILER ON)
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    set(HAS_HIP ON)
    set(FOUND_DEVICE_COMPILER ON)
endif()

if(NOT FOUND_DEVICE_COMPILER)
    message(FATAL_ERROR "Requires at least one device device compiler")
endif()

#TEMPORARY FIX AS THROUGH PRESETS IT DOES NOT WORK
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(sources)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/experiments)
if(BASELINER_HAS_CUDA)
  set(sources cuda/NothingKernel.cu)
endif()

if(BASELINER_HAS_HIP)
  set(sources hip/NothingKernel.hip)
endif()

add_executable(nothing-kernel-bin ${sources})
target_link_libraries(nothing-kernel-bin PRIVATE baseliner::baseliner)


