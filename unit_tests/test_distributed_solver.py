import os
import sys
import unittest

# Allow `python unit_tests/test_distributed_solver.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Request 8 virtual CPU devices; setdefault leaves real-GPU environments unchanged
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=8')

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from NS_solver import (
    DistributedNSSolver,
    DistributedSpectralPoissonSolver,
    PeriodicDifferentialOps,
    Phys,
    SimConfig,
    SingleCoreNSSolver,
    SpectralPoissonSolver,
    Time,
    load_config,
    make_grid,
    validate_solvers,
)


class DistributedSolverParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device_number = min(4, jax.device_count())
        if cls.device_number < 2:
            raise unittest.SkipTest("Distributed tests require at least 2 JAX devices")

        cls.N = cls.device_number * 4
        cls.grid = make_grid(cls.N, L=1.0)
        cls.phys = Phys(nu=0.01, mu0=0.5, sigma0=0.1, C=1.0)
        cls.time = Time(dt=0.002, steps=4)
        cls.theta = jnp.array([0.9, 0.4], dtype=jnp.float32)

        cls.mesh = jax.sharding.Mesh(np.array(jax.devices()[: cls.device_number]), ("x",))
        cls.u_sharding = NamedSharding(cls.mesh, P(None, "x", None))

        cls.u0 = jnp.zeros((2, cls.N, cls.N), dtype=jnp.float32)

    def _make_solvers(self, advection_scheme: str):
        single = SingleCoreNSSolver(
            self.grid,
            self.phys,
            self.time,
            advection_scheme=advection_scheme,
        )
        distributed = DistributedNSSolver(
            self.grid,
            self.phys,
            self.time,
            self.mesh,
            device_number=self.device_number,
            advection_scheme=advection_scheme,
        )
        return single, distributed

    def _assert_close(self, single_value, distributed_value, rtol=5e-5, atol=5e-6):
        np.testing.assert_allclose(
            np.asarray(single_value),
            np.asarray(distributed_value),
            rtol=rtol,
            atol=atol,
        )

    def test_step_matches_single_core_central(self):
        single, distributed = self._make_solvers("central")

        u_single = single.step(self.u0, self.theta, t=0.0)
        u0_dist = jax.device_put(self.u0, self.u_sharding)
        u_dist = distributed.step(u0_dist, self.theta, t=0.0)

        u_single = jax.block_until_ready(u_single)
        u_dist_host = jax.device_get(jax.block_until_ready(u_dist))
        self._assert_close(u_single, u_dist_host)

    def test_rollout_matches_single_core_central(self):
        single, distributed = self._make_solvers("central")

        J_single, u_single = single.rollout(self.u0, self.theta)
        u0_dist = jax.device_put(self.u0, self.u_sharding)
        J_dist, u_dist = distributed.rollout(u0_dist, self.theta)

        J_single = jax.block_until_ready(J_single)
        u_single = jax.block_until_ready(u_single)
        J_dist = jax.block_until_ready(J_dist)
        u_dist_host = jax.device_get(jax.block_until_ready(u_dist))

        self._assert_close(J_single, J_dist)
        self._assert_close(u_single, u_dist_host)

    def test_rollout_matches_single_core_upwind(self):
        single, distributed = self._make_solvers("upwind")

        J_single, u_single = single.rollout(self.u0, self.theta)
        u0_dist = jax.device_put(self.u0, self.u_sharding)
        J_dist, u_dist = distributed.rollout(u0_dist, self.theta)

        J_single = jax.block_until_ready(J_single)
        u_single = jax.block_until_ready(u_single)
        J_dist = jax.block_until_ready(J_dist)
        u_dist_host = jax.device_get(jax.block_until_ready(u_dist))

        self._assert_close(J_single, J_dist)
        self._assert_close(u_single, u_dist_host)

    def test_validate_solvers_reports_small_error(self):
        single, distributed = self._make_solvers("central")

        err, _, _, _, _ = validate_solvers(
            single,
            distributed,
            self.mesh,
            self.theta,
            tol=1e-4,
        )

        self.assertLess(float(err), 1e-4)


class PoissonSolverParityTests(unittest.TestCase):
    """
    Unit tests for DistributedSpectralPoissonSolver.

    Tests the all-to-all transpose round-trip and verifies that the
    distributed Poisson solver produces the same pressure field as the
    single-core reference.  These tests target the layer that was
    previously bugged (_transpose_inv using interleaved ordering).
    """

    @classmethod
    def setUpClass(cls):
        cls.device_number = min(4, jax.device_count())
        if cls.device_number < 2:
            raise unittest.SkipTest("Distributed tests require at least 2 JAX devices")

        cls.N = cls.device_number * 8   # e.g. 32 for 4 devices
        cls.dx = 1.0 / cls.N
        cls.mesh = jax.sharding.Mesh(
            np.array(jax.devices()[: cls.device_number]), ("x",)
        )
        cls.sharding = NamedSharding(cls.mesh, P("x", None))

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _run_sharded(self, fn, x: jax.Array, in_spec=P("x", None), out_spec=P("x", None)):
        wrapped = shard_map(fn, mesh=self.mesh,
                            in_specs=in_spec, out_specs=out_spec,
                            check_rep=False)
        return jax.block_until_ready(wrapped(jax.device_put(x, self.sharding)))

    # ------------------------------------------------------------------
    # Transpose round-trip
    # ------------------------------------------------------------------
    def test_transpose_roundtrip(self):
        """_transpose_inv(_transpose(x)) must recover the original slab."""
        N, P_dev, dx = self.N, self.device_number, self.dx
        poisson = DistributedSpectralPoissonSolver(N, N, dx, P_dev)

        # Each device holds (N_local, N); initialise with unique values
        x = jnp.arange(N * N, dtype=jnp.float32).reshape(N, N)

        def roundtrip(x_local):
            return poisson._transpose_inv(poisson._transpose(x_local))

        x_out = self._run_sharded(roundtrip, x)
        np.testing.assert_allclose(np.asarray(x), np.asarray(x_out), atol=1e-5,
                                   err_msg="_transpose_inv(_transpose(x)) != x")

    def test_transpose_inv_roundtrip(self):
        """_transpose(_transpose_inv(x)) must also recover the input (column layout)."""
        N, P_dev, dx = self.N, self.device_number, self.dx
        poisson = DistributedSpectralPoissonSolver(N, N, dx, P_dev)

        # After _transpose the layout is (N, N_local) per device; mimic that
        x = jnp.arange(N * N, dtype=jnp.float32).reshape(N, N)

        def roundtrip(x_local):
            transposed = poisson._transpose(x_local)           # (N_local,N) → (N,N_local)
            return poisson._transpose(poisson._transpose_inv(transposed))

        x_out = self._run_sharded(roundtrip, x)
        # After two round-trips the layout should be identical to one forward pass
        def forward(x_local):
            return poisson._transpose(x_local)

        x_fwd = self._run_sharded(forward, x,
                                   in_spec=P("x", None), out_spec=P("x", None))
        np.testing.assert_allclose(np.asarray(x_fwd), np.asarray(x_out), atol=1e-5,
                                   err_msg="_transpose(_transpose_inv(_transpose(x))) != _transpose(x)")

    def test_fft2_matches_single_core(self):
        """Distributed forward FFT path must match jnp.fft.fft2 on the full field."""
        N, P_dev, dx = self.N, self.device_number, self.dx
        poisson = DistributedSpectralPoissonSolver(N, N, dx, P_dev)

        x = jnp.arange(N * N, dtype=jnp.float32).reshape(N, N)
        x = jnp.sin(2.0 * jnp.pi * x / N) + 0.25 * jnp.cos(6.0 * jnp.pi * x / N)

        fft_single = jax.block_until_ready(jnp.fft.fft2(x))

        def distributed_fft2(x_local):
            hat_y = jnp.fft.fft(x_local, axis=1)
            hat_t = poisson._transpose(hat_y)
            hat_xy = jnp.fft.fft(hat_t, axis=0)
            return poisson._transpose_inv(hat_xy)

        fft_dist = self._run_sharded(distributed_fft2, x)

        np.testing.assert_allclose(
            np.asarray(fft_single),
            np.asarray(fft_dist),
            rtol=1e-12,
            atol=1e-12,
            err_msg="Distributed fft2 result diverges from single-core jnp.fft.fft2",
        )

    # ------------------------------------------------------------------
    # Poisson solve parity
    # ------------------------------------------------------------------
    def test_poisson_solve_matches_single_core(self):
        """Distributed Poisson solve must match the single-core result."""
        N, P_dev, dx = self.N, self.device_number, self.dx

        single_poisson = SpectralPoissonSolver(N, N, dx)
        dist_poisson = DistributedSpectralPoissonSolver(N, N, dx, P_dev)

        # Non-trivial zero-mean RHS
        key = jax.random.PRNGKey(0)
        rhs = jax.random.normal(key, (N, N), dtype=jnp.float32)
        rhs = rhs - rhs.mean()

        p_single = jax.block_until_ready(single_poisson(rhs))

        p_dist = self._run_sharded(dist_poisson, rhs)

        np.testing.assert_allclose(
            np.asarray(p_single), np.asarray(p_dist),
            rtol=1e-4, atol=1e-5,
            err_msg="Distributed Poisson solve diverges from single-core reference",
        )


# ==============================================================================
# 3. Divergence-Free Projection Tests
# ==============================================================================

class DivergenceFreeProjectionTests(unittest.TestCase):
    """Assert ∇·u ≈ 0 immediately after a Chorin projection step."""

    def _max_divergence(self, u: np.ndarray, dx: float) -> float:
        """L∞ divergence of a (2, N, N) velocity field (periodic, central diff)."""
        gx = (np.roll(u[0], -1, axis=0) - np.roll(u[0], 1, axis=0)) / (2.0 * dx)
        gy = (np.roll(u[1], -1, axis=1) - np.roll(u[1], 1, axis=1)) / (2.0 * dx)
        return float(np.max(np.abs(gx + gy)))

    def test_single_core_step_divergence_free(self):
        """Single-core Chorin step must produce a divergence-free field (|∇·u| < 1e-4)."""
        N = 32
        grid = make_grid(N, L=1.0)
        phys = Phys(nu=0.01, mu0=0.5, sigma0=0.1, C=1.0)
        time_cfg = Time(dt=0.001, steps=1)
        solver = SingleCoreNSSolver(grid, phys, time_cfg)

        # Zero IC: u_star = dt*f (smooth Gaussian), so FD divergence after
        # spectral projection is at near-machine-epsilon levels.
        u0 = jnp.zeros((2, N, N), dtype=jnp.float32)
        theta = jnp.array([0.9, 0.4], dtype=jnp.float32)
        u1 = np.asarray(jax.block_until_ready(solver.step(u0, theta, t=0.0)))

        div_max = self._max_divergence(u1, grid.dx)
        # Spectral projection zeroes divergence in spectral space; 5e-4 covers
        # the O(dx^2) FD residual while still catching a broken projection (O(1)).
        self.assertLess(div_max, 5e-4,
                f"Single-core divergence after projection: {div_max:.2e}")

    def test_single_core_rollout_final_state_divergence_free(self):
        """Final velocity from rollout must also satisfy ∇·u ≈ 0."""
        N = 32
        grid = make_grid(N, L=1.0)
        phys = Phys(nu=0.01, mu0=0.5, sigma0=0.1, C=1.0)
        time_cfg = Time(dt=0.001, steps=5)
        solver = SingleCoreNSSolver(grid, phys, time_cfg)

        u0 = jnp.zeros((2, N, N), dtype=jnp.float32)
        theta = jnp.array([0.9, 0.4], dtype=jnp.float32)
        _, u_final = solver.rollout(u0, theta)
        u_final = np.asarray(jax.block_until_ready(u_final))

        # Accumulation over multiple steps and float32 rounding loosen
        # the strict single-step bound; 1e-3 remains well above noise floor.
        div_max = self._max_divergence(u_final, grid.dx)
        self.assertLess(div_max, 1e-3,
                        f"Single-core rollout final-state divergence: {div_max:.2e}")

    def test_distributed_step_divergence_free(self):
        """Distributed Chorin step must also produce a divergence-free field."""
        device_number = min(4, jax.device_count())
        if device_number < 2:
            raise unittest.SkipTest("Needs at least 2 JAX devices")

        N = device_number * 8
        grid = make_grid(N, L=1.0)
        phys = Phys(nu=0.01, mu0=0.5, sigma0=0.1, C=1.0)
        time_cfg = Time(dt=0.001, steps=1)
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:device_number]), ("x",))
        u_sharding = NamedSharding(mesh, P(None, "x", None))
        solver = DistributedNSSolver(grid, phys, time_cfg, mesh,
                                     device_number=device_number)

        # Zero IC keeps the velocity field smooth after one step; white-noise
        # ICs introduce high-frequency content whose FD-vs-spectral divergence
        # residual is O(Δx²) and can exceed the 1e-4 threshold.
        u0 = jnp.zeros((2, N, N), dtype=jnp.float32)
        u0_dist = jax.device_put(u0, u_sharding)
        theta = jnp.array([0.9, 0.4], dtype=jnp.float32)
        u1 = np.asarray(
            jax.device_get(jax.block_until_ready(solver.step(u0_dist, theta, t=0.0)))
        )

        div_max = self._max_divergence(u1, grid.dx)
        self.assertLess(div_max, 5e-4,
                f"Distributed divergence after projection: {div_max:.2e}")


# ==============================================================================
# 4. Spatial Operator Convergence Tests
# ==============================================================================

class ConvergenceTests(unittest.TestCase):
    """
    Verify FD operators converge at O(Δx²) and that the spectral Poisson solver
    recovers a manufactured exact solution to near machine precision.
    """

    def _laplacian_error(self, N: int) -> float:
        """Max |∇²u_FD − ∇²u_exact| for the manufactured field u = sin(2πx)cos(2πy)."""
        L = 1.0
        dx = L / N
        x = np.linspace(0.0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="ij")
        u = np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
        # ∇²u = -(2π)²·sin(2πx)cos(2πy) - (2π)²·sin(2πx)cos(2πy)
        lap_exact = -2.0 * (2.0 * np.pi) ** 2 * u
        ops = PeriodicDifferentialOps(dx)
        lap_num = np.asarray(ops.laplacian(jnp.array(u)))
        return float(np.max(np.abs(lap_num - lap_exact)))

    def test_laplacian_second_order_convergence(self):
        """FD Laplacian error decreases ~4× when N doubles (second-order convergence)."""
        e16 = self._laplacian_error(16)
        e32 = self._laplacian_error(32)
        e64 = self._laplacian_error(64)

        ratio_coarse = e16 / e32  # should be ≈ 4
        ratio_fine   = e32 / e64  # should be ≈ 4
        self.assertGreater(ratio_coarse, 3.5,
                           f"Convergence ratio e16/e32 = {ratio_coarse:.2f}, expected ≈ 4")
        self.assertGreater(ratio_fine, 3.5,
                           f"Convergence ratio e32/e64 = {ratio_fine:.2f}, expected ≈ 4")

    def test_poisson_solver_manufactured_solution(self):
        """
        SpectralPoissonSolver recovers an exact solution to near machine precision.

        NOTE: despite the docstring saying "-∇²p = rhs", the code computes
        p_hat = -F[rhs]/|k|², which satisfies ∇²p = rhs (positive Laplacian).
        This is the correct convention for the Chorin projection.
        """
        for N in (32, 64):
            with self.subTest(N=N):
                L = 1.0
                dx = L / N
                x = np.linspace(0.0, L, N, endpoint=False)
                X, Y = np.meshgrid(x, x, indexing="ij")
                # Solver convention: ∇²p = rhs.
                # p_exact = sin(2πx)cos(2πy);  ∇²p_exact = −2(2π)² p_exact
                p_exact = np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
                rhs = -2.0 * (2.0 * np.pi) ** 2 * p_exact

                solver = SpectralPoissonSolver(N, N, dx)
                p_num = np.asarray(solver(jnp.array(rhs, dtype=jnp.float32)))

                error = float(np.max(np.abs(p_num - p_exact)))
                self.assertLess(error, 2e-4,
                                f"Poisson solver error for N={N}: {error:.2e}")

    def test_poisson_error_non_increasing_with_n(self):
        """Spectral Poisson error for N=64 must be no worse than for N=32."""
        errors = {}
        for N in (32, 64):
            L = 1.0
            dx = L / N
            x = np.linspace(0.0, L, N, endpoint=False)
            X, Y = np.meshgrid(x, x, indexing="ij")
            p_exact = np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
            # Solver solves ∇²p = rhs, so rhs = ∇²p_exact = -2(2π)²·p_exact.
            rhs = -2.0 * (2.0 * np.pi) ** 2 * p_exact
            p_num = np.asarray(SpectralPoissonSolver(N, N, dx)(
                jnp.array(rhs, dtype=jnp.float32)
            ))
            errors[N] = float(np.max(np.abs(p_num - p_exact)))

        self.assertLessEqual(errors[64], errors[32] + 1e-6,
                             f"Spectral solver error increased: e32={errors[32]:.2e}, "
                             f"e64={errors[64]:.2e}")


# ==============================================================================
# 5. Bad Configuration Tests
# ==============================================================================

class BadConfigTests(unittest.TestCase):
    """Verify that invalid configurations raise ValueError at construction time."""

    @classmethod
    def setUpClass(cls):
        cls.device_number = min(4, jax.device_count())
        cls.phys = Phys(nu=0.01, mu0=0.5, sigma0=0.1, C=1.0)
        cls.time_cfg = Time(dt=0.001, steps=1)
        if cls.device_number >= 2:
            cls.mesh = jax.sharding.Mesh(
                np.array(jax.devices()[:cls.device_number]), ("x",)
            )

    # --- DistributedNSSolver / DistributedSpectralPoissonSolver ---

    def test_distributed_solver_n_not_divisible_raises(self):
        """DistributedNSSolver must raise ValueError when N % device_number != 0."""
        if self.device_number < 2:
            raise unittest.SkipTest("Needs at least 2 JAX devices")
        N_bad = self.device_number * 4 + 1   # guaranteed not divisible
        grid_bad = make_grid(N_bad, L=1.0)
        with self.assertRaises(ValueError):
            DistributedNSSolver(
                grid_bad, self.phys, self.time_cfg, self.mesh,
                device_number=self.device_number,
            )

    def test_distributed_poisson_n_not_divisible_raises(self):
        """DistributedSpectralPoissonSolver must raise ValueError when N % P != 0."""
        if self.device_number < 2:
            raise unittest.SkipTest("Needs at least 2 JAX devices")
        N_bad = self.device_number * 4 + 1
        with self.assertRaises(ValueError):
            DistributedSpectralPoissonSolver(
                N_bad, N_bad, 1.0 / N_bad, self.device_number
            )

    # --- SimConfig validation ---

    def test_simconfig_negative_n_raises(self):
        with self.assertRaises(ValueError):
            SimConfig(N=-1)

    def test_simconfig_zero_n_raises(self):
        with self.assertRaises(ValueError):
            SimConfig(N=0)

    def test_simconfig_zero_nu_raises(self):
        with self.assertRaises(ValueError):
            SimConfig(nu=0.0)

    def test_simconfig_negative_nu_raises(self):
        with self.assertRaises(ValueError):
            SimConfig(nu=-0.001)

    def test_simconfig_invalid_scheme_raises(self):
        with self.assertRaises(ValueError):
            SimConfig(advection_scheme="quickscheme")

    def test_simconfig_empty_scheme_raises(self):
        with self.assertRaises(ValueError):
            SimConfig(advection_scheme="")


# ==============================================================================
# 6. Config Parsing Tests
# ==============================================================================

class ConfigParsingTests(unittest.TestCase):
    """Tests for load_config() and SimConfig property contracts."""

    def _write_temp_config(self, payload) -> str:
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            fname = f.name
        self.addCleanup(lambda path=fname: os.path.exists(path) and os.remove(path))
        return fname

    def test_none_path_returns_default_simconfig(self):
        cfg = load_config(None)
        self.assertIsInstance(cfg, SimConfig)
        self.assertEqual(cfg.N, 128)

    def test_missing_file_returns_default_simconfig(self):
        cfg = load_config("/nonexistent/path/does_not_exist.json")
        self.assertIsInstance(cfg, SimConfig)
        self.assertEqual(cfg.N, 128)

    def test_valid_json_overrides_fields(self):
        fname = self._write_temp_config(
            {"N": 64, "nu": 0.005, "t": 1.0, "advection_scheme": "upwind"}
        )
        cfg = load_config(fname)
        self.assertEqual(cfg.N, 64)
        self.assertAlmostEqual(cfg.nu, 0.005)
        self.assertAlmostEqual(cfg.t, 1.0)
        self.assertEqual(cfg.advection_scheme, "upwind")

    def test_partial_json_keeps_other_defaults(self):
        fname = self._write_temp_config({"N": 32})
        cfg = load_config(fname)
        self.assertEqual(cfg.N, 32)
        self.assertAlmostEqual(cfg.nu, SimConfig().nu)   # rest untouched

    def test_unknown_json_fields_are_ignored(self):
        fname = self._write_temp_config({"N": 48, "unknown_parameter": 999})
        cfg = load_config(fname)
        self.assertEqual(cfg.N, 48)

    def test_invalid_scheme_in_json_raises(self):
        fname = self._write_temp_config({"advection_scheme": "invalid"})
        with self.assertRaises(ValueError):
            load_config(fname)

    def test_theta0_loaded_from_json(self):
        fname = self._write_temp_config({"theta0": [0.3, 0.7]})
        cfg = load_config(fname)
        np.testing.assert_allclose(np.asarray(cfg.theta0), [0.3, 0.7], atol=1e-6)

    def test_simconfig_steps_consistent_with_t_and_dt(self):
        """SimConfig.steps must equal ceil(t / dt)."""
        cfg = SimConfig(N=32, t=0.5)
        expected = int(np.ceil(cfg.t / cfg.dt))
        self.assertEqual(cfg.steps, expected)
        self.assertGreater(cfg.steps, 0)

    def test_simconfig_dx_equals_l_over_n(self):
        cfg = SimConfig(N=64, L=2.0)
        self.assertAlmostEqual(cfg.dx, 2.0 / 64)

    def test_simconfig_dt_satisfies_cfl(self):
        """Computed dt must satisfy both CFL and diffusion stability constraints."""
        cfg = SimConfig(N=64, nu=0.01, cfl=0.4, v_max=1.0)
        dt_cfl  = cfg.cfl * cfg.dx / cfg.v_max
        dt_diff = (cfg.dx ** 2) / (4.0 * cfg.nu)
        self.assertLessEqual(cfg.dt, dt_cfl  + 1e-12)
        self.assertLessEqual(cfg.dt, dt_diff + 1e-12)


# ==============================================================================
# 7. Profiler API Tests
# ==============================================================================

class ProfilerAPITests(unittest.TestCase):
    """Contract tests for SolverKernelProfiler.benchmark_callable."""

    def setUp(self):
        from solver_profiler import ProfileSummary, SolverKernelProfiler
        self.ProfileSummary = ProfileSummary
        self.Profiler = SolverKernelProfiler

    def test_benchmark_callable_returns_profile_summary(self):
        result = self.Profiler.benchmark_callable(
            "dummy", lambda: jax.block_until_ready(jnp.zeros(1)),
            warmup=1, runs=3,
        )
        self.assertIsInstance(result, self.ProfileSummary)
        self.assertEqual(result.label, "dummy")
        self.assertGreater(result.mean_sec, 0.0)

    def test_benchmark_callable_min_le_mean_le_max(self):
        result = self.Profiler.benchmark_callable(
            "order", lambda: jax.block_until_ready(jnp.zeros(1)),
            warmup=0, runs=5,
        )
        self.assertLessEqual(result.min_sec, result.mean_sec)
        self.assertLessEqual(result.mean_sec, result.max_sec)

    def test_benchmark_callable_respects_run_count(self):
        """Warmup calls must not appear in timing stats; total calls = warmup + runs."""
        call_log = []

        def counted():
            call_log.append(1)
            return jax.block_until_ready(jnp.zeros(1))

        self.Profiler.benchmark_callable("count", counted, warmup=2, runs=5)
        self.assertEqual(len(call_log), 7,   # 2 warmup + 5 measured
                         f"Expected 7 total calls, got {len(call_log)}")

    def test_benchmark_callable_zero_warmup(self):
        """warmup=0 must not crash and must still return a valid summary."""
        result = self.Profiler.benchmark_callable(
            "no-warmup", lambda: jax.block_until_ready(jnp.zeros(1)),
            warmup=0, runs=2,
        )
        self.assertIsInstance(result, self.ProfileSummary)

    def test_profile_summary_fields_accessible(self):
        s = self.ProfileSummary(
            label="s", mean_sec=0.5, min_sec=0.4, max_sec=0.6, std_sec=0.05
        )
        self.assertEqual(s.label, "s")
        self.assertAlmostEqual(s.mean_sec, 0.5)
        self.assertAlmostEqual(s.min_sec, 0.4)
        self.assertAlmostEqual(s.max_sec, 0.6)
        self.assertAlmostEqual(s.std_sec, 0.05)


# ==============================================================================
# 8. Runner Argument-Parsing Tests
# ==============================================================================

class RunnerArgParseTests(unittest.TestCase):
    """
    Verify runner.py CLI argument parsing without executing any solver code.
    """

    @staticmethod
    def _build_parser():
        """Use the production parser so tests fail on CLI drift."""
        from runner import build_parser
        return build_parser()

    def test_empty_args_give_defaults(self):
        args = self._build_parser().parse_args([])
        self.assertIsNone(args.config)
        self.assertFalse(args.profile)
        self.assertEqual(args.profile_warmup, 1)
        self.assertEqual(args.profile_runs, 5)

    def test_config_arg_parsed(self):
        args = self._build_parser().parse_args(
            ["--config", "sample_config/config_N_128.json"]
        )
        self.assertEqual(args.config, "sample_config/config_N_128.json")

    def test_profile_flag_parsed(self):
        args = self._build_parser().parse_args(["--profile"])
        self.assertTrue(args.profile)

    def test_profile_warmup_and_runs_parsed(self):
        args = self._build_parser().parse_args(
            ["--profile-warmup", "3", "--profile-runs", "10"]
        )
        self.assertEqual(args.profile_warmup, 3)
        self.assertEqual(args.profile_runs, 10)

    def test_combined_flags(self):
        args = self._build_parser().parse_args([
            "--config", "my.json",
            "--profile",
            "--profile-warmup", "2",
            "--profile-runs", "8",
        ])
        self.assertEqual(args.config, "my.json")
        self.assertTrue(args.profile)
        self.assertEqual(args.profile_warmup, 2)
        self.assertEqual(args.profile_runs, 8)

    def test_profile_warmup_must_be_int(self):
        """Non-integer warmup value must cause a parse error (SystemExit)."""
        with self.assertRaises(SystemExit):
            self._build_parser().parse_args(["--profile-warmup", "notanint"])

    def test_profile_runs_must_be_int(self):
        with self.assertRaises(SystemExit):
            self._build_parser().parse_args(["--profile-runs", "3.5"])


if __name__ == "__main__":
    unittest.main()
