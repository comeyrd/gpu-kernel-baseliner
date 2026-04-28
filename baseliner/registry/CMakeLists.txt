
target_sources(baseliner PRIVATE Components.cpp StorageManager.cpp Register.cpp)
if(BASELINER_HAS_CUDA)
    target_sources(baseliner PRIVATE cuda/CudaRegister.cu)
endif()
if(BASELINER_HAS_HIP)
       target_sources(baseliner PRIVATE hip/HipRegister.hip)
endif()
