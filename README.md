# Baseliner

GPU benchmarking library for C++ with native support for CUDA and HIP. Provides statistically rigorous kernel timing through configurable stopping criteria and reproducible protocol files.

> ⚠️ Research software under active development. Interfaces may change.

## Building

Requires CMake 3.15+, C++17, and at least one backend:
- CUDA 11.0+ (12.0+ recommended)
- HIP 5.2+ (7.0+ recommended)

```bash
cmake -S . -B build -DBASELINER_BUILD_EXAMPLES=ON
cmake --build build
```

The binary is located at `build/bin/baseliner-example` after building with examples enabled.

## CLI Reference

### `gen` - Generate Configuration Files

| Command | Description | Output |
|---------|-------------|--------|
| `gen --metadata` | Lists all registered components (workloads, backends, criteria, stats) | `metadata.json` |
| `gen --schema` | JSON schema for protocol file validation | `protocol.schema.json` |
| `gen --default-pf` | Protocol file with all default values for registered workloads | `default-protocol.json` |
| `gen --minimal-pf` | Minimal protocol file template | `minimal-protocol.json` |
| `gen --saved-pf` | Protocol file from saved recipes (if any) | `saved-protocol.json` |

**Example:**
```bash
./build/bin/baseliner-example gen --default-pf my-protocol.json
```

### `run` - Execute Benchmarks

| Option | Description | Example |
|--------|-------------|---------|
| `--protocol-files` / `-pf` | Run benchmarks from one or more protocol files | `run -pf bench.json` |
| `--output-file` | Specify output JSON path (default: `result-<uuid>.json`) | `run -pf bench.json --output-file out.json` |
| `--replay-runs` / `-rr` | Replay protocol from previous result files | `run -rr result.json` |
| `--nvbench` | Use NVBench-style defaults (no protocol file needed) | `run --nvbench` |
| `--primbench` | Use PrimBench-style defaults (no protocol file needed) | `run --primbench` |
| `--tiny` | Quick debug run with minimal iterations | `run --tiny` |
| `--device` | Select GPU device when using default modes | `run --nvbench --device 1` |

**Examples:**
```bash
# Run from protocol file
./build/bin/baseliner-example run --protocol-files protocol.json

# Multiple protocol files
./build/bin/baseliner-example run -pf bench1.json bench2.json

# Quick test with NVBench defaults
./build/bin/baseliner-example run --nvbench --tiny
```

## Protocol Files

Protocol files define reproducible benchmark configurations in JSON format.

### Structure

```json
{
  "baseliner_version": "0.9.0",
  "presets": { /* component configurations */ },
  "stats_presets": { /* statistics configurations */ },
  "recipes": { /* stopping criteria + stats combinations */ },
  "campaigns": [ /* workload + backend + recipe mappings */ ]
}
```

### Presets

Configure individual components. Each component (workload, backend, stopping criterion, benchmark) has a preset with options.

**Example - Workload preset:**
```json
"MatrixMul": {
  "default": {
    "description": "Default preset",
    "options": {
      "MatrixMulWorkload": {
        "block_size": { "value": "32" },
        "wA": { "value": "1024" },
        "hA": { "value": "1024" },
        "wB": { "value": "1024" }
      },
      "Workload": {
        "work_size": { "value": "50" },
        "seed": { "value": "333" }
      }
    }
  }
}
```

**Example - Backend preset:**
```json
"cuda": {
  "default": {
    "options": {
      "Backend": {
        "device": { "value": "0" },
        "lock_clock": { "value": "0" }
      }
    }
  }
}
```

**Example - Stopping criterion preset:**
```json
"EntropyStoppingCriterion": {
  "default": {
    "options": {
      "EntropySC": {
        "min_samples": { "value": "10" },
        "max_angle": { "value": "0.048" }
      },
      "StoppingCriterion": {
        "max_nb_repetition": { "value": "2000" }
      }
    }
  }
}
```

**Example - Benchmark preset:**
```json
"Benchmark": {
  "default": {
    "options": {
      "Benchmark": {
        "warmup": { "value": "1" },
        "flush": { "value": "0" },
        "batch_size": { "value": "25" },
        "dynamic_batch": { "value": "1" },
        "block": { "value": "0" },
        "warm_cool": { "value": "0" }
      }
    }
  }
}
```

### Recipes

Combine benchmark settings, stopping criterion, and statistics into named configurations.

```json
"recipes": {
  "fast": {
    "description": "Quick benchmark with fixed iterations",
    "benchmark": {
      "impl": "Benchmark",
      "preset": "default"
    },
    "stopping": {
      "impl": "StoppingCriterion",
      "preset": "default"
    },
    "stats": {
      "preset": "default"
    }
  },
  "accurate": {
    "description": "Precise benchmark with entropy convergence",
    "benchmark": {
      "impl": "Benchmark",
      "preset": "default"
    },
    "stopping": {
      "impl": "EntropyStoppingCriterion",
      "preset": "default"
    },
    "stats": {
      "preset": "default"
    }
  }
}
```

### Campaigns

Map workloads to backends and recipes. Multiple campaigns can run different configurations.

```json
"campaigns": [
  {
    "name": "matmul-benchmark",
    "recipe": "accurate",
    "workloads": [
      {
        "impl": "MatrixMul",
        "preset": "default"
      }
    ],
    "backends": [
      {
        "impl": "cuda",
        "preset": "default"
      }
    ],
    "on_incompatible": "Skip"
  }
]
```

**Multiple backends example:**
```json
"backends": [
  { "impl": "cuda", "preset": "default" },
  { "impl": "hip", "preset": "default" }
]
```

### Parameter Sweeps

Define parameter ranges to explore. Sweeps generate multiple benchmark runs with different configurations.

**Cartesian sweep (all combinations):**
```json
"workloads": [
  {
    "impl": "MatrixMul",
    "preset": "default",
    "sweep": {
      "strategy": "Cartesian",
      "axes": [
        {
          "name": "MatrixMulWorkload.block_size",
          "values": ["16", "32"]
        },
        {
          "name": "MatrixMulWorkload.wA",
          "values": ["512", "1024", "2048"]
        }
      ]
    }
  }
]
```
This generates 2 × 3 = 6 runs.

**Zip sweep (parallel iteration):**
```json
"sweep": {
  "strategy": "Zip",
  "axes": [
    {
      "name": "MatrixMulWorkload.wA",
      "values": ["512", "1024", "2048"]
    },
    {
      "name": "MatrixMulWorkload.hA",
      "values": ["512", "1024", "2048"]
    }
  ]
}
```
This generates 3 runs: (512,512), (1024,1024), (2048,2048).

## Stopping Criteria

| Criterion | Description | Key Options |
|-----------|-------------|-------------|
| `StoppingCriterion` | Fixed iteration count | `max_nb_repetition` |
| `EntropyStoppingCriterion` | Stop when entropy stabilizes (NVBench-style) | `min_samples`, `max_angle`, `min_r2` |
| `StdRelStoppingCriterion` | Stop when relative stddev stabilizes (PrimBench-style) | `max_noise`, `min_samples`, `noise_stability_threshold` |
| `ConfidenceIntervalMedianSC` | Stop when confidence interval narrow enough | `precision`, `relative_error_th` |
| `VariationStoppingCriterion` | Stop after duration with noise tolerance | `min_duration_ms`, `noise_tolerance` |

## Benchmark Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `warmup` | bool | Run one warmup iteration before timing | `1` |
| `batch_size` | int | Number of kernel launches per batch | `25` |
| `dynamic_batch` | bool | Automatically adjust batch size for efficiency | `1` |
| `flush` | bool | Flush L2 cache between iterations (cold cache) | `0` |
| `block` | bool | Use blocking kernel to prevent overlap | `0` |
| `block_duration` | float | Duration of blocking kernel in ms | `1000.0` |
| `warm_cool` | bool | Actively manage GPU temperature | `0` |
| `min_gpu_temp` | float | Minimum GPU temperature (°C) | `50.0` |
| `max_gpu_temp` | float | Maximum GPU temperature (°C) | `60.0` |
| `validate_workload` | bool | Validate kernel results after execution | `0` |

## Output Format

Results are written as JSON containing:

- **Hardware info** — GPU name, compute capability, driver version
- **Measurements** — raw timing samples, batch statistics
- **Derived metrics** — median, mean, stddev, throughput, arithmetic intensity
- **Sweep point** — parameter configuration for this run
- **Metadata** — Baseliner version, stopping criterion used

**Example output structure:**
```json
{
  "baseliner_version": "0.9.0",
  "hardware": { "device_name": "...", "compute_capability": "..." },
  "results": [
    {
      "sweep_point": { "MatrixMulWorkload.block_size": "32" },
      "measurements": [
        { "name": "Median", "value": 1.234, "unit": "ms" },
        { "name": "FLOPThroughput", "value": 5678.9, "unit": "GFLOP/s" }
      ]
    }
  ]
}
```

## Examples

The repository includes two example workloads in `examples/`:
- **MatMulWorkload** — tiled matrix multiplication
- **ComputationWorkload** — vector arithmetic

Build with examples enabled and run:
```bash
# Generate default protocol
./build/bin/baseliner-example gen --default-pf protocol.json

# Run benchmarks
./build/bin/baseliner-example run -pf protocol.json
```

## For Developers

To implement custom workloads, stopping criteria, or statistics for Baseliner, see [ARCHITECTURE.md](ARCHITECTURE.md).