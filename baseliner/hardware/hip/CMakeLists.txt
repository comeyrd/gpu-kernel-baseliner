target_sources(baseliner PRIVATE
    HipBackend.hip
    BlockingKernel.hip
    L2Flusher.hip
    HipStats.hip
    Timer.hip)

if(BASELINER_HAS_NVML)
    target_link_libraries(baseliner PUBLIC AMD::amdsmi)
    target_sources(baseliner PRIVATE AmdSmiUtils.hip)
endif()
