# Architecture

Baseliner uses a registry-based plugin system where workloads, backends, stopping criteria, and statistics are registered at startup via macros and instantiated by the orchestrator at runtime.

## Component System

### Backends

Backends abstract hardware operations (streams, events, memory, timing) behind a common interface. CUDA and HIP backends are included.

The backend interface defines:
- Stream management
- Event creation and timing
- Memory operations
- Device queries
- Clock locking (optional)
- Temperature monitoring (optional, via NVML/AMD SMI)

Adding support for a new framework requires implementing `IBackend<T>`.

### Workload Lifecycle

The measurement loop distinguishes six workload stages:

1. **Host setup** — allocate and initialize host memory
2. **Device allocation** — allocate device memory, transfer data
3. **Device reset** — reset state between iterations
4. **Kernel execution** — launch and time the kernel
5. **Result fetch** — copy results back to host
6. **Cleanup** — free resources

Each stage can be individually timed. Only execution is measured by default for throughput calculations.

### Stopping Criteria

Trials terminate when a stopping criterion is satisfied. The criterion receives timing samples and decides when to stop.

Included implementations:
- **Fixed count** — stop after N iterations
- **Entropy convergence** — based on NVBench (measures information gain per sample)
- **Coefficient of variation** — based on PrimBench (relative standard deviation threshold)
- **Confidence interval** — stop when CI width falls below threshold

Custom criteria implement `StoppingCriterion` and register via `BASELINER_REGISTER_STOPPING_CRITERION`.

### Statistics Engine

A dependency graph tracks relationships between measurements and derived metrics.

**Flow:**
1. Raw samples (execution time) are collected
2. Batch statistics computed per iteration
3. Element statistics computed across all samples
4. Derived metrics computed from dependencies

**Example dependencies:**
- `ExecutionTime` (raw) → `Median`, `Mean`, `StdDev`
- `Median` + `FLOPCount` → `FLOPThroughput`
- `FLOPThroughput` + `DataThroughput` → `ArithmeticIntensity`

Stats implement `IStats` and register via `BASELINER_REGISTER_STAT`.

### Registry System

Components register themselves at static initialization:

```cpp
namespace {
  BASELINER_REGISTER_WORKLOAD(MyWorkload);
  BASELINER_REGISTER_STOPPING_CRITERION(MyCriterion);
  BASELINER_REGISTER_STAT(MyStat);
  BASELINER_REGISTER_BACKEND("mybackend", MyBackend);
}
```

The orchestrator queries the registry by name and instantiates components at runtime based on protocol file specifications.

## Writing a Workload

Workloads inherit from `IWorkload<BackendT>` and implement lifecycle methods:

```cpp
template <typename BackendT>
class MyKernel : public Baseliner::IWorkload<BackendT> {
  using backend = BackendT;

  // Called once at trial start
  void setup_host_random_generated() override {
    // Initialize host data
  }

  // Called once per trial, before timing loop
  void setup_device(typename backend::stream_t stream) override {
    // Allocate device memory
    // Copy data to device
  }

  // Called before each timed iteration
  void reset_device(typename backend::stream_t stream) override {
    // Reset device state if needed (e.g., zero output)
  }

  // The timed kernel launch
  auto run(typename backend::stream_t stream)
      -> typename backend::launch_result_t override {
    // Launch kernel
    return {};  // or return cooperative_launch_result for CUDA
  }

  // Called once after timing loop
  void fetch_results(typename backend::stream_t stream) override {
    // Copy results back to host
    // Free device memory
  }

  // Called at trial end
  void free() override {
    // Final cleanup if needed
  }

  // Optional: define metrics
  auto number_of_floating_point_operations() -> std::optional<size_t> override {
    return m_flops;
  }

  auto number_of_bytes() -> std::optional<size_t> override {
    return m_bytes;
  }
};
```

### Backend Specialization

Define backend-specific implementations via template specialization:

```cpp
// In .cu file
template <>
void MyKernel<CudaBackend>::setup_device(cudaStream_t stream) {
  CHECK_CUDA(cudaMalloc(&d_ptr, size));
  CHECK_CUDA(cudaMemcpyAsync(d_ptr, h_ptr, size, cudaMemcpyHostToDevice, stream));
}

template <>
auto MyKernel<CudaBackend>::run(cudaStream_t stream) -> std::monostate {
  my_cuda_kernel<<<grid, block, 0, stream>>>(d_ptr, N);
  return {};
}

namespace {
  BASELINER_REGISTER_WORKLOAD(MyKernel<CudaBackend>);
}
```

```cpp
// In .hip file
template <>
void MyKernel<HipBackend>::setup_device(hipStream_t stream) {
  CHECK_HIP(hipMalloc(&d_ptr, size));
  CHECK_HIP(hipMemcpyAsync(d_ptr, h_ptr, size, hipMemcpyHostToDevice, stream));
}

template <>
auto MyKernel<HipBackend>::run(hipStream_t stream) -> std::monostate {
  hipLaunchKernelGGL(my_hip_kernel, grid, block, 0, stream, d_ptr, N);
  return {};
}

namespace {
  BASELINER_REGISTER_WORKLOAD(MyKernel<HipBackend>);
}
```

### Options System

Workloads can expose configuration options via the options system:

```cpp
class MyKernel : public IWorkload<BackendT> {
  void register_options() override {
    IWorkload<BackendT>::register_options();  // Call parent
    this->add_option("MyKernel", "tile_size", "Tile size for computation", m_tile_size);
    this->add_option("MyKernel", "use_shared_mem", "Enable shared memory", m_use_shared);
  }

private:
  int m_tile_size = 16;
  bool m_use_shared = true;
};
```

These options become available in protocol files under the workload preset.

## Benchmark Loop

The `Benchmark` class orchestrates the measurement:

1. Apply backend options (device selection, clock locking)
2. Generate sweep points (if parameter sweep defined)
3. For each sweep point:
   - Apply sweep point options
   - Create stream and timer
   - Setup host data
   - Setup device
   - Warmup run (optional)
   - Loop until stopping criterion satisfied:
     - Thermal management (optional)
     - Blocking kernel (optional)
     - Batch of timed runs
     - Compute statistics
     - Dynamic batch adjustment (optional)
   - Fetch results
   - Validate (optional)
   - Cleanup

All configurable via protocol file or benchmark options.

## Accuracy Features

### L2 Cache Flushing

Enabled via `flush: true` in benchmark options. Allocates a large buffer and streams through it between iterations to evict L2 cache contents. Ensures cold-cache timing but adds overhead.

### Stream Blocking

Enabled via `block: true`. Launches a long-running blocking kernel that holds the GPU. Unblocks periodically to let benchmark iterations execute. Prevents kernel overlap and improves timing consistency.

### Thermal Management

Enabled via `warm_cool: true`. Monitors GPU temperature and:
- Warms GPU by running compute kernels if below `min_gpu_temp`
- Cools GPU by idling if above `max_gpu_temp`
- Waits for temperature to stabilize before starting batch

Requires NVML (NVIDIA) or AMD SMI (AMD) support.

### Dynamic Batching

Enabled via `dynamic_batch: true`. Adjusts batch size dynamically based on batch duration:
- If batch duration < `minimal_batch_duration`, double batch size
- If batch duration > 2× `minimal_batch_duration`, halve batch size

Reduces overhead for very fast kernels while keeping measurement time reasonable.

## Statistics

Built-in statistics include:
- **ExecutionTime** — raw kernel timing samples
- **Median**, **Mean**, **StdDev**, **Min**, **Max** — aggregates
- **BatchSize**, **BatchTime** — batch metadata
- **FLOPThroughput**, **DataThroughput** — derived from workload metrics
- **ArithmeticIntensity** — FLOP/byte ratio
- **DeviceTemperature** — GPU temperature monitoring
- **CpuTime**, **WarmupTime**, **SetupTime** — lifecycle timings

Custom statistics can compute arbitrary derived values from dependencies.
