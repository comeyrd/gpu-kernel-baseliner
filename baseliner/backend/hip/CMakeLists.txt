target_sources(baseliner PRIVATE
    HipBackend.hip
    BlockingKernel.hip
    L2Flusher.hip
    Timer.hip
    HipStats.hip)

if(BASELINER_HAS_AMDSMI)
    target_link_libraries(baseliner PUBLIC AMD::amdsmi)
    target_sources(baseliner PRIVATE AmdSmiUtils.hip)
endif()