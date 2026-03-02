target_sources(baseliner PRIVATE
    HipBackend.hip
    BlockingKernel.hip
    L2Flusher.hip
    HipStats.hip
    Timer.hip)

if(BASELINER_HAS_NVML)
    target_link_libraries(baseliner PUBLIC HIP::nvml)
    target_sources(baseliner PRIVATE NvmlUtils.hip)
endif()
