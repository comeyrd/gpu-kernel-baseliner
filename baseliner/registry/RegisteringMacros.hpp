#ifndef BASELINER_REGISTRY_REGISTERINGMACROS_HPP
#define BASELINER_REGISTRY_REGISTERINGMACROS_HPP
#include <baseliner/registry/Registrars.hpp>

#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif

#define BASELINER_CONCAT_IMPL(a, b) a##b
#define BASELINER_CONCAT(a, b) BASELINER_CONCAT_IMPL(a, b)
#define BASELINER_UNIQUE_NAME(base) BASELINER_CONCAT(base, __LINE__)

#define INNER_BASELINER_REGISTER_STAT(Stat)                                                                            \
  ATTRIBUTE_USED const static Baseliner::GeneralStatRegistrar<Stat> BASELINER_UNIQUE_NAME(_registrar_Stat_){#Stat};

#define INNER_BASELINER_REGISTER_STOPPING_CRITERION(Stopping)                                                          \
  ATTRIBUTE_USED const static Baseliner::StoppingRegistrar<Stopping> BASELINER_UNIQUE_NAME(_registrar_Stopping_){      \
      #Stopping};

#define INNER_BASELINER_REGISTER_DEFAULT_STATS(DefaultStats)                                                           \
  ATTRIBUTE_USED const static Baseliner::StatConceptRegistrar BASELINER_UNIQUE_NAME(_registrar_DefaultStat_){          \
      DefaultStats};

#define INNER_BASELINER_REGISTER_BENCHMARK(Benchmark)                                                                  \
  ATTRIBUTE_USED const static Baseliner::BenchmarkRegistrar<Benchmark> BASELINER_UNIQUE_NAME(_registrar_Benchmark_){   \
      #Benchmark};

#define INNER_BASELINER_REGISTER_BACKEND(name, Backend)                                                                \
  ATTRIBUTE_USED const static Baseliner::BackendRegistrar<Backend> BASELINER_UNIQUE_NAME(_registrar_Backend_){name};

#define INNER_BASELINER_REGISTER_WORKLOAD(Workload)                                                                    \
  ATTRIBUTE_USED const static Baseliner::WorkloadRegistrar<Workload> BASELINER_UNIQUE_NAME(_registrar_Workload_){      \
      #Workload};

#define INNER_BASELINER_REGISTER_WORKLOAD_NAME(Workload, name)                                                         \
  ATTRIBUTE_USED const static Baseliner::WorkloadRegistrar<Workload> BASELINER_UNIQUE_NAME(_registrar_Workload_){name};

#define INNER_BASELINER_REGISTER_KERNEL(Kernel)                                                                        \
  ATTRIBUTE_USED const static Baseliner::KernelRegistrar<Kernel> BASELINER_UNIQUE_NAME(_registrar_Kernel_){#Kernel};

#define INNER_BASELINER_REGISTER_BACKEND_STATS(Stat)                                                                   \
  ATTRIBUTE_USED const static Baseliner::BackendStatRegistrar<Stat> BASELINER_UNIQUE_NAME(_registrar_BackendStat_){    \
      #Stat};

#endif // BASELINER_REGISTRY_REGISTERINGMACROS_HPP