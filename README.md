# Differentiable Navier–Stokes Solver (JAX, Single-Core & Distributed)

## Overview

This project implements a modular, differentiable 2D incompressible Navier–Stokes solver using Chorin’s projection method on a periodic domain.

Rather than a standalone script, the solver is designed as a **composable simulation system**, where core numerical components (differential operators, advection schemes, Poisson solvers) are decoupled and interchangeable.

The implementation emphasizes:
- **Modularity**: clean separation of numerical components
- **Parallel scalability**: support for both single-device and multi-device execution
- **Differentiability**: end-to-end compatibility with JAX automatic differentiation
- **System integration**: designed to serve as a building block for ML-driven simulation workflows

---

## System Design

The solver is structured into modular components:

- **DifferentialOps**
  - Encapsulates spatial operators: gradient, divergence, Laplacian
  - Supports both single-core and distributed implementations

- **AdvectionScheme**
  - Strategy pattern for nonlinear term discretization
  - Supports central differencing and upwind schemes

- **PoissonSolver**
  - Spectral FFT-based solver for pressure projection
  - Includes both single-device and distributed variants

- **ChorinProjector**
  - Core time-stepping logic (advection → pressure solve → projection)
  - Independent of parallelization strategy

- **Solver Layer**
  - Provides rollout, objective evaluation, and gradient computation
  - Unified interface for single-core and distributed execution

This separation allows numerical methods and execution strategies to be modified without changing the overall workflow.

## Project Layout

- `NS_solver.py`: core solver library with numerical operators, Poisson solvers, forcing, and solver classes
- `runner.py`: CLI entry point for validation, profiling, and visualization generation
- `solver_profiler.py`: stage-wise benchmarking utilities for solver kernels
- `studies.py`: standalone research studies for viscosity sweeps, gradient benchmarking, scheme comparison, and optimizer comparison
- `unit_tests/`: parity and regression tests for distributed rollout, FFT, transpose, and Poisson solve behavior
- `sample_config/`: example JSON configurations for different grid sizes
- `result/`: generated plots, NumPy arrays, and `.npz` study outputs

---

## Numerical Method

Solve the incompressible Navier–Stokes equations:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}$$

$$\nabla \cdot \mathbf{u} = 0$$

using **Chorin’s projection method**:

1. Advection-diffusion (explicit Euler)
2. Pressure Poisson solve (spectral FFT)
3. Projection to divergence-free velocity field

The domain is periodic on $[0, L]^2$.

---

## Parallelization Strategy

The solver uses **dynamic device detection**: at runtime, it checks the number of available JAX devices and automatically switches between single-core and distributed implementations.

- **Single device (1)**: Uses `PeriodicDifferentialOps` with `jnp.roll` for periodic wrapping
- **Multiple devices (N > 1)**: Uses `DistributedDifferentialOps` with domain decomposition and halo exchange

### Distributed execution (when N > 1 devices)

When running with multiple devices, the solver uses **SPMD parallelism via JAX `shard_map`**.

Key techniques:

- **Domain decomposition**
  - Partition along x-direction into equal slabs (one per device)

- **Halo exchange**
  - Implemented using `jax.lax.ppermute`
  - Ensures correct stencil computation across device boundaries

- **Distributed FFT**
  - Uses `all-to-all` communication to perform global spectral solve
  - Sequence:
    ```
    FFT_y → all-to-all → FFT_x → solve → iFFT_x → all-to-all⁻¹ → iFFT_y
    ```

- **Global reductions**
  - Objective accumulated using `lax.psum`

This design enables scaling across devices while preserving the same solver abstraction.

---

## Differentiability

The solver is fully differentiable with respect to input parameters using JAX automatic differentiation.

This enables:
- parameter inference
- optimization
- integration with ML models (e.g., differentiable physics)

The system exposes:
- JIT-compiled objective function
- JIT-compiled gradients via reverse-mode AD

---

## Data Flow & API

The solver provides a clean API:

```python
J, u_final = solver.rollout(u0, theta)
grad = solver.jit_gradient(theta)
```

This allows it to be used as:

- a standalone simulation tool
- a component in optimization loops
- a backend for ML pipelines

## Validation

The implementation is validated through:

- Single vs distributed consistency checks
- Gradient benchmarking
   - JAX automatic differentiation vs finite differences
- Parameter studies
   - viscosity regimes
   - advection schemes

These ensure numerical correctness and stability across configurations.

## Testing

The project includes a comprehensive unit test suite in `unit_tests/test_distributed_solver.py` using Python's `unittest` framework.

### Test Structure

The suite currently covers 44 tests across parity, projection correctness, manufactured-solution checks, configuration validation, and CLI/profiling API behavior.

**DistributedSolverParityTests** — Validates distributed rollout and stepping against the single-core reference:
- `test_step_matches_single_core_central` — Single projection step with central advection
- `test_rollout_matches_single_core_central` — Full rollout parity with central advection
- `test_rollout_matches_single_core_upwind` — Full rollout parity with upwind advection
- `test_validate_solvers_reports_small_error` — Verifies the solver-comparison helper stays within tolerance

**PoissonSolverParityTests** — Tests the distributed spectral Poisson solver and FFT communication path:
- `test_transpose_roundtrip` — Forward/inverse all-to-all transpose round-trip correctness
- `test_transpose_inv_roundtrip` — Reverse transpose round-trip on column-layout data
- `test_fft2_matches_single_core` — Distributed 2D FFT path matches `jnp.fft.fft2`
- `test_poisson_solve_matches_single_core` — Distributed pressure solve matches single-core reference

**DivergenceFreeProjectionTests** — Checks the core incompressibility contract after projection:
- Single-core one-step projection produces a near divergence-free state
- Single-core rollout final state remains near divergence-free
- Distributed one-step projection produces a near divergence-free state

**ConvergenceTests** — Verifies expected numerical behavior as the grid is refined:
- Finite-difference Laplacian shows second-order convergence under grid refinement
- Spectral Poisson solve matches a manufactured exact solution
- Manufactured-solution Poisson error is non-increasing from `N=32` to `N=64`

**BadConfigTests** — Confirms invalid configurations fail fast:
- `DistributedNSSolver` and `DistributedSpectralPoissonSolver` reject `N % device_number != 0`
- `SimConfig` rejects invalid `N`, invalid `nu`, and unsupported advection schemes

**ConfigParsingTests** — Covers user JSON parsing and `SimConfig` property behavior:
- Missing config path and missing files fall back to defaults
- Partial JSON overrides only supplied fields
- Unknown JSON fields are ignored
- Invalid user JSON values still raise through `SimConfig`
- `theta0`, `steps`, `dx`, and `dt` behavior is validated explicitly

**ProfilerAPITests** — Exercises the profiling helper contract:
- `benchmark_callable()` returns a populated `ProfileSummary`
- Timing statistics obey `min <= mean <= max`
- Warmup and measured-run counts are applied as expected

**RunnerArgParseTests** — Checks CLI argument handling without running the full solver:
- Default argument values are stable
- `--config`, `--profile`, `--profile-warmup`, and `--profile-runs` parse correctly
- Invalid integer arguments fail with parser errors

### Running Tests

Run all tests:
```bash
python -m unittest unit_tests.test_distributed_solver
```

Run a specific test class:
```bash
python -m unittest unit_tests.test_distributed_solver.DistributedSolverParityTests
```

Run a specific test:
```bash
python -m unittest unit_tests.test_distributed_solver.DistributedSolverParityTests.test_rollout_matches_single_core_central
```

### Test Configuration

Test bootstrap is defined at the top of `unit_tests/test_distributed_solver.py`:
- Sets default virtual device count to 8 using `XLA_FLAGS`
- Configures `sys.path` for proper module imports
- Runs before `import jax`, ensuring distributed tests execute under `unittest`

The parity tests use relative tolerance (`rtol=5e-5`) and absolute tolerance (`atol=5e-6`) to account for floating-point rounding differences between single-core and distributed execution paths. Projection and manufactured-solution tests use problem-specific thresholds appropriate to the discretization and float32 execution.

## Studies & Experiments

The project includes several studies:

- Viscosity sweep
   - explores flow regimes and stability
- Gradient benchmarking
   - validates AD correctness and performance
- Advection scheme comparison
   - central vs upwind discretization
- Optimization study
   - parameter optimization using Optax

The standalone studies entry point in `studies.py` currently runs four workflows:

- `ViscosityStudy`: scans viscosity values and records objective, kinetic energy, velocity statistics, and vorticity-derived plots
- `GradientBenchmark`: compares JAX reverse-mode AD against finite-difference estimates across multiple parameter vectors and epsilon values
- `SchemeComparisonStudy`: compares central and upwind advection across multiple viscosity regimes
- `ParameterOptimizationStudy`: compares Optax optimizers over a fixed optimization budget

## Usage

### Docker Reproducibility

The repository now includes a CPU-focused Docker setup so other users can reproduce the environment without manually installing Python or JAX dependencies.

### Build the image

```bash
docker build -t ns-solver-jax .
```

### Run the default solver command

This uses the image default command:

```bash
docker run --rm -it \
  -v "$(pwd)/result:/app/result" \
  ns-solver-jax
```

### Run with a specific config

```bash
docker run --rm -it \
  -e XLA_FLAGS='--xla_force_host_platform_device_count=8' \
  -v "$(pwd)/result:/app/result" \
  ns-solver-jax \
  python runner.py --config sample_config/config_N_512.json
```

### Run tests in Docker

```bash
docker run --rm -it ns-solver-jax \
  python -m unittest unit_tests.test_distributed_solver
```

### Use Docker Compose

The compose file mounts `result/` back to the host so generated plots and arrays are preserved locally.

```bash
docker compose up --build
```

Override the virtual JAX device count at runtime:

```bash
JAX_DEVICE_COUNT=1 docker compose run --rm ns-solver
JAX_DEVICE_COUNT=8 docker compose run --rm ns-solver
```

### Notes

- The container is CPU-only for portability and reproducibility.
- `MPLBACKEND=Agg` is set so plotting works in headless environments.
- Distributed behavior is reproduced through host-side virtual devices via `XLA_FLAGS`, matching the project’s current testing approach.

### Device Configuration

The solver dynamically adapts to the number of virtual devices available at runtime. Control this via the `XLA_FLAGS` environment variable before invoking the script.

Use `runner.py` as the primary CLI entry point. (`NS_solver.py` forwards to the same runner.)

#### Single-core (1 device) — True single-core baseline
```bash
XLA_FLAGS='--xla_force_host_platform_device_count=1' python runner.py --config sample_config/config_N_128.json
```

#### Distributed (8 devices) — Multi-device SPMD execution
```bash
XLA_FLAGS='--xla_force_host_platform_device_count=8' python runner.py --config sample_config/config_N_128.json
```

**Note:** `DEVICE_NUMBER` is now a runtime value derived from `jax.device_count()`. In practice, the selected `XLA_FLAGS` setting controls how many virtual host devices JAX exposes.

Runner behavior by device count:
- `jax.device_count() == 1`: runs single-core solver only; distributed comparison and stage-wise comparison profiling are skipped.
- `jax.device_count() > 1`: runs full single-core vs distributed validation/comparison and distributed profiling.

### Run solver with default configuration
```bash
python runner.py
```

### Run solver with specified config
```bash
python runner.py --config sample_config/config_N_1024.json
```

### Run solver through the dedicated runner
```bash
python runner.py --config sample_config/config_N_1024.json
```

Equivalent compatibility entry point:
```bash
python NS_solver.py --config sample_config/config_N_1024.json
```

### Run with profiling
```bash
# Single-core profiling
XLA_FLAGS='--xla_force_host_platform_device_count=1' python runner.py --config sample_config/config_N_128.json --profile --profile-warmup 1 --profile-runs 5

# Distributed profiling
XLA_FLAGS='--xla_force_host_platform_device_count=8' python runner.py --config sample_config/config_N_128.json --profile --profile-warmup 1 --profile-runs 5
```

### Run studies
```bash
python studies.py
```

Run studies with a specific configuration:
```bash
python studies.py --config sample_config/config_N_128.json
```

The studies driver uses the current `SimConfig` API and executes end-to-end with the current codebase. It writes plots and `.npz` summaries into `result/`, including:

- `result/viscosity_study_results.npz`
- `result/gradient_benchmark_results.npz`
- `result/scheme_comparison_results.npz`
- `result/optimization_study_results.npz`

The main solver runner also writes comparison artifacts such as:

- `result/velocity_magnitude_distributed_P{device_count}_N{N}.png`
- `result/velocity_magnitude_distributed_P{device_count}_N{N}.npy`

For single-device runs, runner writes:

- `result/velocity_magnitude_single_N{N}.png`
- `result/velocity_magnitude_single_N{N}.npy`

## System Assumptions
- Periodic boundary conditions
- Uniform square grid
- Grid size divisible by number of virtual devices (set via `XLA_FLAGS` at runtime)
- Explicit time stepping (CFL-constrained)
- Environment: XLA simulated multi-device on host (via `--xla_force_host_platform_device_count`)

## Design Philosophy

This project is not intended as a production CFD solver, but as a research software system with:

- clear modular abstractions
- support for experimentation
- compatibility with ML workflows
- scalable parallel execution

It is designed to bridge:

numerical simulation ↔ machine learning systems

## Future Extensions
- Non-periodic boundary conditions
- Adaptive mesh refinement
- GPU multi-device support
- Integration with learned surrogate models

## References
- Chorin projection method
- Spectral methods for Poisson equation
- JAX documentation: https://jax.readthedocs.io/
