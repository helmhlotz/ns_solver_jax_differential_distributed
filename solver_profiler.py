import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P



@dataclass(frozen=True)
class ProfileSummary:
    label: str
    mean_sec: float
    min_sec: float
    max_sec: float
    std_sec: float


class SolverKernelProfiler:
    """Profile single-core and distributed solver kernels."""

    def __init__(self, mesh: jax.sharding.Mesh, logger: Optional[logging.Logger] = None):
        self.mesh = mesh
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    @staticmethod
    def _block_tree(x: Any) -> Any:
        """Block until a pytree of JAX arrays is fully computed."""
        return jax.tree_util.tree_map(jax.block_until_ready, x)

    @classmethod
    def benchmark_callable(
        cls,
        label: str,
        fn: Callable,
        *args,
        warmup: int = 1,
        runs: int = 5,
    ) -> ProfileSummary:
        """Benchmark a callable that returns JAX arrays/scalars."""
        for _ in range(max(0, warmup)):
            cls._block_tree(fn(*args))

        samples: List[float] = []
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            cls._block_tree(fn(*args))
            samples.append(time.perf_counter() - t0)

        arr = np.array(samples, dtype=np.float64)
        return ProfileSummary(
            label=label,
            mean_sec=float(arr.mean()),
            min_sec=float(arr.min()),
            max_sec=float(arr.max()),
            std_sec=float(arr.std()),
        )

    def _log_profile_table(self, title: str, rows: List[ProfileSummary]) -> None:
        self.logger.info("-" * 80)
        self.logger.info(title)
        self.logger.info("-" * 80)
        self.logger.info(f"{'Kernel':36s} {'mean [s]':>12s} {'min [s]':>12s} {'max [s]':>12s} {'std [s]':>12s}")
        for row in rows:
            self.logger.info(
                f"{row.label:36s} {row.mean_sec:12.6f} {row.min_sec:12.6f} {row.max_sec:12.6f} {row.std_sec:12.6f}"
            )

    def profile_solver_kernels(
        self,
        single,
        distributed,
        theta: jax.Array,
        warmup: int = 1,
        runs: int = 5,
    ) -> None:
        """Profile solver stages for both single-core and distributed implementations."""
        mesh = self.mesh
        N = single.grid.N
        dt = single.time.dt
        u_sharding = NamedSharding(mesh, P(None, "x", None))
        u0_single = jnp.zeros((2, N, N))
        u0_dist = jax.device_put(jnp.zeros((2, N, N)), u_sharding)
        t0 = jnp.array(0.0)

        # ---------------- Single-core kernels ----------------
        s_ops = single.projector.ops
        s_adv = single.projector.advection
        s_poi = single.projector.poisson
        s_force = single.forcing
        s_nu = single.projector.nu

        @jax.jit
        def s_forcing(theta_, t_):
            return s_force(theta_, t_)

        @jax.jit
        def s_advec_diff(u_, f_):
            ux, uy = u_[0], u_[1]
            lap_ux = s_ops.laplacian(ux)
            lap_uy = s_ops.laplacian(uy)
            adv_ux = s_adv.advect(ux, ux, uy, s_ops)
            adv_uy = s_adv.advect(uy, ux, uy, s_ops)
            ux_star = ux + dt * (s_nu * lap_ux + adv_ux + f_[0])
            uy_star = uy + dt * (s_nu * lap_uy + adv_uy + f_[1])
            return jnp.stack([ux_star, uy_star], axis=0)

        @jax.jit
        def s_pressure(u_star_):
            return s_poi(s_ops.divergence(u_star_) / dt)

        @jax.jit
        def s_projection(u_star_, p_):
            px, py = s_ops.grad(p_)
            return jnp.stack([u_star_[0] - dt * px, u_star_[1] - dt * py], axis=0)

        s_step = jax.jit(lambda u_, theta_, t_: single.step(u_, theta_, t_))
        s_rollout = jax.jit(lambda u_, theta_: single.rollout(u_, theta_))

        f_s = s_forcing(theta, t0)
        ustar_s = s_advec_diff(u0_single, f_s)
        p_s = s_pressure(ustar_s)

        single_rows = [
            self.benchmark_callable("single: forcing", s_forcing, theta, t0, warmup=warmup, runs=runs),
            self.benchmark_callable("single: advection_diffusion", s_advec_diff, u0_single, f_s, warmup=warmup, runs=runs),
            self.benchmark_callable("single: pressure_solve", s_pressure, ustar_s, warmup=warmup, runs=runs),
            self.benchmark_callable("single: projection", s_projection, ustar_s, p_s, warmup=warmup, runs=runs),
            self.benchmark_callable("single: full_step", s_step, u0_single, theta, t0, warmup=warmup, runs=runs),
            self.benchmark_callable(
                "single: full_rollout",
                s_rollout,
                u0_single,
                theta,
                warmup=warmup,
                runs=max(1, min(runs, 3)),
            ),
        ]

        # ---------------- Distributed kernels ----------------
        d_ops = distributed.projector.ops
        d_adv = distributed.projector.advection
        d_poi = distributed.projector.poisson
        d_force = distributed.forcing
        d_nu = distributed.projector.nu
        d_grid = distributed.grid
        d_device_number = distributed.device_number

        def forcing_local_kernel(theta_, t_):
            my_id = jax.lax.axis_index("x")
            rows = d_grid.N // d_device_number
            return d_force.local_field(theta_, t_, row_start=my_id * rows, rows=rows)

        d_forcing_local = jax.jit(
            shard_map(
                forcing_local_kernel,
                mesh=mesh,
                in_specs=(P(None), P()),
                out_specs=P(None, "x", None),
                check_rep=False,
            )
        )

        def advec_diff_local_kernel(u_local_, f_local_):
            ux, uy = u_local_[0], u_local_[1]
            lap_u, adv_u = d_ops.laplacian_and_advect_vec(u_local_, d_adv)
            ux_star = ux + dt * (d_nu * lap_u[0] + adv_u[0] + f_local_[0])
            uy_star = uy + dt * (d_nu * lap_u[1] + adv_u[1] + f_local_[1])
            return jnp.stack([ux_star, uy_star], axis=0)

        d_advec_diff = jax.jit(
            shard_map(
                advec_diff_local_kernel,
                mesh=mesh,
                in_specs=(P(None, "x", None), P(None, "x", None)),
                out_specs=P(None, "x", None),
                check_rep=False,
            )
        )

        def pressure_local_kernel(u_star_local_):
            return d_poi(d_ops.divergence_vec(u_star_local_) / dt)

        d_pressure = jax.jit(
            shard_map(
                pressure_local_kernel,
                mesh=mesh,
                in_specs=(P(None, "x", None),),
                out_specs=P("x", None),
                check_rep=False,
            )
        )

        def projection_local_kernel(u_star_local_, p_local_):
            px, py = d_ops.grad(p_local_)
            return jnp.stack([u_star_local_[0] - dt * px, u_star_local_[1] - dt * py], axis=0)

        d_projection = jax.jit(
            shard_map(
                projection_local_kernel,
                mesh=mesh,
                in_specs=(P(None, "x", None), P("x", None)),
                out_specs=P(None, "x", None),
                check_rep=False,
            )
        )

        d_step = jax.jit(distributed._step_sharded)
        d_rollout = jax.jit(lambda u_, theta_: distributed.rollout(u_, theta_))

        f_d = d_forcing_local(theta, t0)
        ustar_d = d_advec_diff(u0_dist, f_d)
        p_d = d_pressure(ustar_d)

        dist_rows = [
            self.benchmark_callable("distributed: forcing_local", d_forcing_local, theta, t0, warmup=warmup, runs=runs),
            self.benchmark_callable("distributed: advection_diffusion", d_advec_diff, u0_dist, f_d, warmup=warmup, runs=runs),
            self.benchmark_callable("distributed: pressure_solve", d_pressure, ustar_d, warmup=warmup, runs=runs),
            self.benchmark_callable("distributed: projection", d_projection, ustar_d, p_d, warmup=warmup, runs=runs),
            self.benchmark_callable("distributed: full_step", d_step, u0_dist, theta, t0, warmup=warmup, runs=runs),
            self.benchmark_callable(
                "distributed: full_rollout",
                d_rollout,
                u0_dist,
                theta,
                warmup=warmup,
                runs=max(1, min(runs, 3)),
            ),
        ]

        self._log_profile_table("PROFILE SUMMARY (single-core)", single_rows)
        self._log_profile_table("PROFILE SUMMARY (distributed)", dist_rows)


def profile_solver_kernels(
    single,
    distributed,
    mesh: jax.sharding.Mesh,
    theta: jax.Array,
    warmup: int = 1,
    runs: int = 5,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Backwards-compatible function wrapper around SolverKernelProfiler."""
    profiler = SolverKernelProfiler(mesh=mesh, logger=logger)
    profiler.profile_solver_kernels(single, distributed, theta, warmup=warmup, runs=runs)
