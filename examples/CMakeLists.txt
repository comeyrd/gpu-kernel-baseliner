if(BASELINER_HAS_CUDA)
  add_subdirectory(cuda)
endif()

if(BASELINER_HAS_HIP)
  add_subdirectory(hip)
endif()
