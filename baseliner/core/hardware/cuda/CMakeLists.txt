target_sources(baseliner_core PRIVATE CudaBackend.cu
    BlockingKernel.cu
    L2Flusher.cu
    CudaStats.cu WarmingGpu.cu)
if(BASELINER_HAS_NVML)
    target_link_libraries(baseliner PUBLIC CUDA::nvml)
    target_sources(baseliner_core PRIVATE NvmlUtils.cu)
endif()