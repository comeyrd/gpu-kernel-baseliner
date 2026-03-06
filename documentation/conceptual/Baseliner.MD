# Conceptual Reference - Baseliner

gpu-kernel-baseliner (refered as Baseliner) is a C++ benchmarking library for gpu kernels (kernels).

In this conceptual reference, we will explain and motivate design choices behind the architecture and all the parts of Baseliner.

## Concept Brief

The Baseliner project aims at providing a library for developers to perform benchmarking on their gpu program.

Its main goal is to provide accurate, reliable and reproducible benchmark results. Theses results would enable developers to derive the right conclusions from a performance analysis.

The other goal is to provide a simple and standard interface to benchmark gpu kernels. This interface would be hardware agnostic, decoupling the hardware-specific code from the benchmarking logic.

### Decoupling Gpu kernels benchmarking

Most of the gpu kernels benchmarks available in the litterature have one shared issue.

Their measurement, benchmarking and kernel logic are all coupled inside the same file.
That makes changing the measuring strategy of a whole benchmark, or the benchmarking strategy of the whole benchmark, a tedious and complicated manual work.

That makes most of the kernel benchmarks available hard to customize and makes the creation of new experiments on the same kernels almost impossible.

To make gpu kernel benchmarks more reusable and

## Benchmarking Gpu kernels

The main part of this library is to do accurate, reliable and reproducible benchmarking of gpu kernels.
