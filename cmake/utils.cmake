# set_if_undefined(<variable> [<value>]...)
#
# Set variable if it is not defined.
macro(set_if_undefined variable)
    if(NOT DEFINED "${variable}")
        set("${variable}" ${ARGN})
    endif()
endmacro()

# win_copy_deps_to_target_dir(<target> [<target-dep>]...)
#
# Creates custom command to copy runtime dependencies to target's directory after building the target.
# Function does nothing if platform is not Windows and ignores all dependencies except shared libraries.
# On CMake 3.21 or newer, function uses TARGET_RUNTIME_DLLS generator expression to obtain list of runtime
# dependencies. Specified dependencies (if any) are still used to find and copy PDB files for debug builds.
function(win_copy_deps_to_target_dir target)
    if(NOT WIN32)
        return()
    endif()

    set(has_runtime_dll_genex NO)

    if(CMAKE_MAJOR_VERSION GREATER 3 OR CMAKE_MINOR_VERSION GREATER_EQUAL 21)
        set(has_runtime_dll_genex YES)

        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -P "${mylib_SOURCE_DIR}/cmake/silent_copy.cmake"
                "$<TARGET_RUNTIME_DLLS:${target}>" "$<TARGET_FILE_DIR:${target}>"
            COMMAND_EXPAND_LISTS)
    endif()

    foreach(dep ${ARGN})
        get_target_property(dep_type ${dep} TYPE)

        if(dep_type STREQUAL "SHARED_LIBRARY")
            if(NOT has_runtime_dll_genex)
                add_custom_command(TARGET ${target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -P "${mylib_SOURCE_DIR}/cmake/silent_copy.cmake" 
                        "$<TARGET_FILE:${dep}>" "$<TARGET_PDB_FILE:${dep}>" "$<TARGET_FILE_DIR:${target}>"
                    COMMAND_EXPAND_LISTS)
            else()
                add_custom_command(TARGET ${target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -P "${mylib_SOURCE_DIR}/cmake/silent_copy.cmake"
                        "$<TARGET_PDB_FILE:${dep}>" "$<TARGET_FILE_DIR:${target}>"
                    COMMAND_EXPAND_LISTS)
            endif()
        endif()
    endforeach()
endfunction()

function(baseliner_enable_git_version TARGET_NAME)
    execute_process(
        COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_REV
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND git diff --quiet
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE IS_DIRTY
    )
    set(DIRTY_STR "")
    if(NOT IS_DIRTY EQUAL 0)
        set(DIRTY_STR "-dirty")
    endif()
    set(FULL_VERSION_STR "${GIT_BRANCH}@${GIT_REV}${DIRTY_STR}")
    target_compile_definitions(${TARGET_NAME} PRIVATE BASELINER_GIT_VERSION="${FULL_VERSION_STR}")
endfunction()