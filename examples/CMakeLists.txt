set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(BASELINER_HAS_CUDA)
  add_subdirectory(cuda)
endif()

if(BASELINER_HAS_HIP)
  add_subdirectory(hip)
endif()
