#ifndef BASELINER_REGISTERING_MACROS_HPP
#define BASELINER_REGISTERING_MACROS_HPP
#include <baseliner/managers/Registrars.hpp>
#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif

#define BASELINER_REGISTER_STAT(Stat)                                                                                  \
  ATTRIBUTE_USED static Baseliner::GeneralStatRegistrar<Stat> _registrar_##Stat{#Stat};

#define BASELINER_REGISTER_SUITE(Suite)                                                                                \
  ATTRIBUTE_USED static Baseliner::SuiteRegistrar<Suite> _registrar_##Suite{#Suite};

#define BASELINER_REGISTER_STOPPING_CRITERION(Stopping)                                                                \
  ATTRIBUTE_USED static Baseliner::StoppingRegistrar<Stopping> _registrar_##Stopping{#Stopping};

#define BASELINER_REGISTER_BENCHMARK(Benchmark)                                                                        \
  ATTRIBUTE_USED static Baseliner::BenchmarkRegistrar<Benchmark> _registrar_##Benchmark{#Benchmark};

#define BASELINER_REGISTER_BACKEND(name, Backend)                                                                      \
  ATTRIBUTE_USED static Baseliner::BackendRegistrar<Backend> _registrar_##Backend{name};

#define BASELINER_REGISTER_CASE(Case) ATTRIBUTE_USED static Baseliner::CaseRegistrar<Case> _registrar_##Case{#Case};

#define BASELINER_REGISTER_KERNEL(Kernel)                                                                              \
  ATTRIBUTE_USED static Baseliner::KernelRegistrar<Kernel> _registrar_##Kernel{#Kernel};

#define BASELINER_REGISTER_BACKEND_STATS(Stat)                                                                         \
  ATTRIBUTE_USED static Baseliner::BackendStatRegistrar<Stat> _registrar_##Stat{#Stat};

#endif // BASELINER_REGISTERING_MACROS_HPP