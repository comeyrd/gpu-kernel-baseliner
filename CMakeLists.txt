cmake_minimum_required(VERSION 3.15)
project(baseliner VERSION 1.0.0 LANGUAGES CXX)

add_library(baseliner SHARED
    src/backend/Backend.hpp
    src/backend/Backend.cpp)

add_library(baseliner::baseliner ALIAS baseliner)

target_include_directories(baseliner PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

option(BASELINER_BUILD_EXAMPLES "Build baseliner examples" OFF)

if(BASELINER_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()