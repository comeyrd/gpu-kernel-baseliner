target_sources(baseliner PRIVATE
    CudaBackend.cu
    BlockingKernel.cu
    L2Flusher.cu
    CudaStats.cu
    Timer.cu)

if(BASELINER_HAS_NVML)
    target_link_libraries(baseliner PUBLIC CUDA::nvml)
    target_sources(baseliner PRIVATE NvmlUtils.cu)
endif()