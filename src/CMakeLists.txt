add_library(baseliner SHARED)
add_library(baseliner::baseliner ALIAS baseliner)
#JSON
include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(json)
target_link_libraries(baseliner PUBLIC nlohmann_json::nlohmann_json)
target_include_directories(baseliner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(backend)
option(BASELINER_BUILD_RESEARCH_QUESTIONS "Build baseliner research questions" OFF)
if(BASELINER_BUILD_RESEARCH_QUESTIONS)
  add_subdirectory(research_questions)
endif()
