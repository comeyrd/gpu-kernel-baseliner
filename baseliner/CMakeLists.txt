add_subdirectory(cli)
add_subdirectory(core)
add_subdirectory(orchestrator)
add_subdirectory(registry)
add_subdirectory(utils)
#JSON


include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG        v3.2
)
FetchContent_MakeAvailable(argparse)

target_link_libraries(baseliner_core PUBLIC nlohmann_json::nlohmann_json argparse)
target_include_directories(baseliner_core PUBLIC 
    # A.From source
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    # B. When installed
    $<INSTALL_INTERFACE:include> 
)


target_link_libraries(baseliner PUBLIC baseliner_core)
target_include_directories(baseliner PUBLIC 
    # A.From source
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    # B. When installed
    $<INSTALL_INTERFACE:include> 
)
