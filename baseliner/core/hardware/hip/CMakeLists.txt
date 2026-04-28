target_sources(baseliner_core PRIVATE
    HipBackend.hip
    BlockingKernel.hip
    L2Flusher.hip
    HipStats.hip
    WarmingGpu.hip
    )

if(BASELINER_HAS_AMDSMI)
    target_link_libraries(baseliner_core PUBLIC AMD::amdsmi)
    target_sources(baseliner_core PRIVATE AmdSmiUtils.hip)
endif()
