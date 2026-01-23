# Gpu Kernel Baseliner

Gpu Kernel Baseliner is a C++ library to produce reliable and accurate execution time measurement for Cuda and Hip Kernels.

**Baseliner** helps developers understand the performance of their kernels in a controlled and easy to use environnement.

This library is still a Work In Progress, the user facing interfaces could be subject to change.

## Use Cases

Baseliner is designed to answer specific performance questions:

- **Regression Testing** : Compare two versions of the same kernel on the same card.
- **Hardware Comparison** : Compare the execution of the same kernel across cards.

## Features

- **Dual Support** : Native support for both **CUDA** and **HIP** runtimes.
- **Statistical Stability** : Handles reruns and numbers of execution to produce reliable measurements.
- **Low Overhead** : Results are the closest possible to the real executions.
- **Minimal Setup & Runtime Tuning:** Designed for easy integration into existing C++ projects. Benchmark parameters can be adjusted at runtime, allowing you to modify test conditions without the need to recompile.

## Requirements

- C++ 17 or higher
- Cmake 3.15+
- **Nvidia**: CUDA Toolkit (11.0+)
- **AMD**: ROCm(5.2+)

_The recommended setup is the highest CUDA or HIP version your device supports._
_Note_: You need at least one of the GPU architectures to compile and use this library.

### Depedencies

- **nlohmann/json**, To simplify saving settings and outputs, we use this json library, it is embedded into the library.

## Installation

**In the futur, the library will be available in binary form**

### Cmake and FetchContent

You can fetch and build the library with the Cmake FetchContent to automatically download the library as a depedency.
Example :

```cmake
FetchContent_Declare(
  baseliner
  GIT_REPOSITORY https://github.com/comeyrd/gpu-kernel-baseliner.git
  GIT_TAG        v0.1
)
FetchContent_MakeAvailable(baseliner)

target_link_libraries(my_benchmark PRIVATE baseliner::baseliner)
```

## Examples

An example of the usage of Baseliner is in the [using-baseliner](https://github.com/comeyrd/using-baseliner) repository.

## Documentation

**In Progress...**

## Contributions

Contributions are welcomed !
There should be some Issues tagged as "getting-started", start with that then we will chat to see what you can contribute on !
