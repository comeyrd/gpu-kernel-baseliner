target_sources(baseliner PRIVATE
    Backend.cpp
)

if(BASELINER_HAS_CUDA)
    add_subdirectory(cuda)
endif()