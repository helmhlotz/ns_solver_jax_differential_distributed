"""
runner.py — CLI entry point for the Navier-Stokes solver.

Owns argument parsing, profiling dispatch, validation, and visualisation.
Core solver logic lives in NS_solver.py; profiling utilities in solver_profiler.py.
"""
import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from NS_solver import (
    DistributedNSSolver,
    Phys,
    SingleCoreNSSolver,
    Time,
    load_config,
    make_grid,
    setup_logger,
    validate_solvers,
)
from solver_profiler import profile_solver_kernels

logger = setup_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the solver entry point."""
    parser = argparse.ArgumentParser(description="Navier-Stokes CFD Solver")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON config file. Missing fields fall back to defaults."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run stage-wise profiling for single-core and distributed solvers."
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=1,
        help="Number of warmup runs per profiled kernel (excluded from statistics)."
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=5,
        help="Number of measured runs per profiled kernel."
    )
    return parser


def main():
    args = build_parser().parse_args()

    logger.info("=" * 80)
    logger.info("NAVIER-STOKES CFD SOLVER")
    logger.info("=" * 80)
    device_number = jax.device_count()
    logger.info(f"JAX Device Count: {device_number}")

    mesh = jax.sharding.Mesh(jax.devices(), ("x",))

    config = load_config(args.config)

    grid = make_grid(config.N, L=config.L)
    phys = Phys(nu=config.nu, mu0=config.mu0, sigma0=config.sigma0, C=config.C_force)
    time_cfg = Time(dt=config.dt, steps=config.steps)
    theta = config.theta0

    logger.info("Building solvers...")
    single_solver = SingleCoreNSSolver(grid, phys, time_cfg, advection_scheme=config.advection_scheme)
    logger.info(f"Advection scheme: {config.advection_scheme}")

    use_distributed_compare = device_number > 1
    u_dist = None
    t_dist = None

    if use_distributed_compare:
        dist_solver = DistributedNSSolver(
            grid,
            phys,
            time_cfg,
            mesh,
            device_number=device_number,
            advection_scheme=config.advection_scheme,
        )

        if args.profile:
            profile_solver_kernels(
                single_solver,
                dist_solver,
                mesh,
                theta,
                warmup=args.profile_warmup,
                runs=args.profile_runs,
                logger=logger,
            )

        # Validation and timing comparison
        _, u_single, u_dist, t_single, t_dist = validate_solvers(
            single_solver, dist_solver, mesh, theta, tol=config.obj_tol
        )
    else:
        logger.info("Single-device runtime detected (jax.device_count() == 1).")
        logger.info("Distributed comparison is disabled; running single-core solver only.")
        if args.profile:
            logger.info("Skipping stage-wise comparison profile because it requires >1 device.")

        t0 = time.perf_counter()
        _, u_single = single_solver.rollout(jnp.zeros((2, config.N, config.N)), theta)
        jax.block_until_ready(u_single)
        t_single = time.perf_counter() - t0
        logger.info(f"Total time used for single-core solver: {t_single:.3f}s")

    # Visualisation
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING VISUALISATION")
    logger.info("=" * 80)
    speed_single = jnp.sqrt(u_single[0] ** 2 + u_single[1] ** 2)

    if use_distributed_compare and u_dist is not None:
        speed_dist = jnp.sqrt(u_dist[0] ** 2 + u_dist[1] ** 2)
        # Use 2nd/98th percentiles so floating-point outliers (O(1e-7) noise)
        # don't compress the colormap into an uninformative band of numerical noise.
        combined = np.concatenate([np.asarray(speed_single).ravel(), np.asarray(speed_dist).ravel()])
    else:
        speed_dist = None
        combined = np.asarray(speed_single).ravel()

    vmin = float(np.percentile(combined, 2))
    vmax = float(np.percentile(combined, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        center = float(np.mean(combined))
        vmin = center - 1e-12
        vmax = center + 1e-12

    extent = [0, config.L, 0, config.L]
    t_label = f"t={config.steps * config.dt:.3f}"

    if use_distributed_compare and speed_dist is not None and t_dist is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        im0 = axes[0].imshow(speed_single.T, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Single-core ({config.advection_scheme}) {t_label}\ntime: {t_single:.3f}s")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(im0, ax=axes[0], label="Speed")

        im1 = axes[1].imshow(speed_dist.T, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Distributed ({device_number} cores, {config.advection_scheme}) {t_label}\ntime: {t_dist:.3f}s")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(im1, ax=axes[1], label="Speed")

        fig.suptitle(f"Velocity Magnitude — Speedup: {t_single / t_dist:.2f}x", fontsize=13)
        save_png = f"result/velocity_magnitude_distributed_P{device_number}_N{config.N}.png"
        save_npy = f"result/velocity_magnitude_distributed_P{device_number}_N{config.N}.npy"
        np.save(save_npy, np.array(speed_dist))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        im0 = ax.imshow(speed_single.T, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Single-core ({config.advection_scheme}) {t_label}\ntime: {t_single:.3f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im0, ax=ax, label="Speed")
        fig.suptitle("Velocity Magnitude — Single-core", fontsize=13)
        save_png = f"result/velocity_magnitude_single_N{config.N}.png"
        save_npy = f"result/velocity_magnitude_single_N{config.N}.npy"
        np.save(save_npy, np.array(speed_single))

    fig.tight_layout()
    os.makedirs("result", exist_ok=True)
    fig.savefig(save_png, dpi=300)
    plt.close(fig)
    logger.info(f"Saved: {save_png}")

    logger.info("=" * 80)
    logger.info("DONE.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
