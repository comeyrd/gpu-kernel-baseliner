# Baseliner

Baseliner is a GPU benchmarking library for C++ that provides statistically rigorous kernel timing with native support for CUDA and HIP.

The library separates measurement logic from kernel code through a plugin architecture where workloads, stopping criteria, and statistics are independently registered components. This design allows the same kernel to run under different measurement strategies without modification.

> ⚠️ This is research software under active development. Interfaces may change.

## Design

### Architecture

Baseliner uses a registry-based plugin system. Workloads, backends, stopping criteria, and statistics are registered at startup via macros and instantiated by the orchestrator at runtime.

The measurement loop distinguishes six workload stages: host setup, device allocation, device reset, kernel execution, result fetch, and cleanup. Each stage can be individually timed.

### Backends

Backends abstract hardware operations (streams, events, memory, timing) behind a common interface. CUDA and HIP backends are included. Adding support for a new framework requires implementing the backend interface.

### Stopping Criteria

Trials terminate when a stopping criterion is satisfied. Included implementations:
- Fixed iteration count
- Entropy convergence (after NVBench)
- Coefficient of variation threshold (after PrimBench)
- Confidence interval width

### Statistics Engine

A dependency graph tracks relationships between measurements and derived metrics. Raw timing samples feed aggregates (median, mean, stddev) which in turn feed derived quantities (throughput, arithmetic intensity). The engine computes values as dependencies resolve.

### Protocol Files

Benchmarks are defined via JSON protocol files specifying presets, recipes, and campaigns. This makes runs reproducible and configuration explicit.

## Building

Requires CMake 3.15+ and C++17. At least one backend compiler is required:
- CUDA 11.0+ (12.0+ recommended)
- HIP 5.2+ (7.0+ recommended)

```bash
cmake -S . -B build -DBASELINER_BUILD_EXAMPLES=ON
cmake --build build
```

The build system detects available compilers and enables corresponding backends. NVML and AMD SMI are optional dependencies used for temperature monitoring.

## Running

The CLI has two modes:

**Generate metadata or protocol files:**
```bash
baseliner-example gen --metadata metadata.json
baseliner-example gen --schema protocol.schema.json
baseliner-example gen --default-pf protocol.json
```

**Run benchmarks:**
```bash
baseliner-example run --protocol-files protocol.json
baseliner-example run --output-file results.json --protocol-files protocol.json
```

Results are written as JSON containing hardware info, measurements, and derived statistics.

## Writing a Workload

Define a workload by inheriting from `IWorkload<Backend>` and implementing the lifecycle methods:

```cpp
template <typename BackendT>
class MyKernel : public Baseliner::IWorkload<BackendT> {
  using backend = BackendT;

  void setup_host_random_generated() override {
    // Initialize host data
  }

  void setup_device(typename backend::stream_t stream) override {
    // Allocate device memory, copy data
  }

  void reset_device(typename backend::stream_t stream) override {
    // Reset state between runs
  }

  auto run(typename backend::stream_t stream)
      -> typename backend::launch_result_t override {
    // Launch kernel
    return {};
  }

  void fetch_results(typename backend::stream_t stream) override {
    // Copy results back, free device memory
  }

  void free() override {}
};
```

Register backend-specific specializations:
```cpp
// In .cu file
template <>
auto MyKernel<CudaBackend>::run(cudaStream_t stream) -> std::monostate {
  my_cuda_kernel<<<grid, block, 0, stream>>>(...);
  return {};
}

namespace {
  BASELINER_REGISTER_WORKLOAD(MyKernel<CudaBackend>);
}
```

See `examples/` for complete implementations.

## Examples

Two workloads are included:
- **MatMulWorkload** — tiled matrix multiply with 16x16 and 32x32 variants
- **ComputationWorkload** — simple vector arithmetic kernel

Both have CUDA and HIP implementations demonstrating the registration pattern.

## Configuration Options

Protocol files support configuration of:
- **Benchmark options** — batching, L2 flushing, thermal management, validation
- **Stopping criteria** — iteration counts, thresholds, confidence levels
- **Statistics** — which metrics to compute and report
- **Sweeps** — parameter ranges to explore (cartesian or zip strategies)

The schema is available via `gen --schema`.

## Requirements

- C++17
- CMake 3.15+
- CUDA 11.0+ or HIP 5.2+
- nlohmann/json (fetched automatically)
- argparse (fetched automatically)

## Performance

Baseliner's GPU event timing matches the precision of vendor tools. Batching and configurable accuracy features (cache flushing, thermal control) allow trading measurement overhead for precision depending on workload requirements.