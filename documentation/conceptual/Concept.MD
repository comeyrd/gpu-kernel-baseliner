# Conceptual Brief - Baseliner

gpu-kernel-baseliner (refered as Baseliner) is a C++ benchmarking library for gpu kernels.
In this conceptual brief, we will explain and motivate design choices behind the architecture and all the parts of Baseliner.

## Concept Brief

The Baseliner project aims at providing a library for developers to perform benchmarking on their gpu program.

The Baseliner design is based on three main goals :

- Its main goal is to provide accurate, reliable and reproducible benchmark results. Theses results would enable developers to derive the right conclusions from a performance analysis.

- The second goal is to provide a simple and standard interface to benchmark gpu kernels. This interface would be hardware agnostic, decoupling the hardware-specific code from the benchmarking logic.

- The third one is to make every part of the Baseliner library customizable,by enable users to write custom implementations of interfaces and connect everything with a plugin system.

### Accurate, reliable and reproducible benchmarking

Providing accurate and reliable results was not a subject tackled by researchers when presenting their benchmarks in the litterature. Most of their results and running were done with ad-hoc scripts, without taking into account many essential specifics about gpu kernel bechmarking.

Baseliner has for goal to define the different parts of benchmarking, and implement for each part the state of the art way of doing benchmarking.

### Simple and Standard GPU Benchmarking Interface

It exists multiple ways to run programs on GPUs, from vendor specific languages and runtimes (Hip, Cuda) to open-standard cross platform languages (Sycl, OpenMP, Triton).
The current objective of baseliner is to actively support Hip, Cuda and Sycl.

But Baseliner won't be hardcoded to these languages, to stay flexible for possible futur languages, it abstracts the hardware specific behind a 'Backend' interface.

Another important part for the standard interface is the decoupling of 'specific' code and 'generic' code.
'Specific' code, as in specific to the targeted hardware or specific to the targeted kernel.
'Generic' code, as in code that would not change if we swap hardware or targeted kernel.

### Customizable by design

To have a reusable gpu kernel benchmarks, users should be able to tweak and test every part of the benchmark, without having to rewrite a lot of code, and it should be as straight forward as changing options and shouldn't need recompilation if not absolutely necessary.

Users should be able to write custom parts for baseliner and easily integrate them, without having to recompile the whole library, only their own code.
