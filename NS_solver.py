import os
import json
import time
import logging
import functools
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding

# Runtime device count is controlled externally via XLA_FLAGS when desired.
DEVICE_NUMBER = jax.device_count()

# ==========================
# Logger
# ==========================
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
    logger.addHandler(handler)
    return logger

logger = setup_logger(__name__)


# ==========================
# Configuration and Data Structures
# ==========================
@functools.partial(jax.tree_util.register_dataclass,
                   data_fields=['theta0'],
                   meta_fields=['N', 'L', 'nu', 'mu0', 'sigma0', 'C_force', 'cfl', 'v_max', 't', 'advection_scheme', 'obj_tol'])
@dataclass(frozen=True)
class SimConfig:
    """Centralized simulation configuration."""
    N: int = 128
    L: float = 1.0
    nu: float = 0.01
    mu0: float = 0.5
    sigma0: float = 0.1
    C_force: float = 1.0
    cfl: float = 0.4 # smaller than 0.5 to ensure stability
    v_max: float = 1.0
    t: float = 0.5
    theta0: Optional[jax.Array] = None
    advection_scheme: str = "central"
    obj_tol: float = 1e-6

    def __post_init__(self):
        if self.N <= 0:
            raise ValueError(f"Grid resolution N must be positive, got {self.N}")
        if self.nu <= 0:
            raise ValueError(f"Kinematic viscosity nu must be positive, got {self.nu}")
        if self.advection_scheme not in ("central", "upwind"):
            raise ValueError(f"Advection scheme must be 'central' or 'upwind', got {self.advection_scheme}")
        if self.theta0 is None:
            object.__setattr__(self, 'theta0', jnp.array([1.0, 0.5]))

    @property
    def steps(self) -> int:
        return int(np.ceil(self.t / self.dt))

    @property
    def dx(self) -> float:
        return self.L / self.N

    @property
    def dt(self) -> float:
        """Stable time step satisfying both CFL and diffusion constraints."""
        dt_advective = self.cfl * self.dx / self.v_max
        dt_diffusive = (self.dx ** 2) / (4.0 * self.nu)
        return float(jnp.minimum(dt_advective, dt_diffusive))
    
    # @property
    # def time(self) -> float:
    #     return float(self.steps * self.dt)


@functools.partial(jax.tree_util.register_dataclass,
    data_fields=['X', 'Y'],
    meta_fields=['N', 'L', 'dx'])
@dataclass(frozen=True)
class Grid:
    """Spatial discretization of the 2D periodic domain."""
    N: int; L: float; dx: float; X: jax.Array; Y: jax.Array


@functools.partial(jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=['nu', 'mu0', 'sigma0', 'C'])
@dataclass(frozen=True)
class Phys:
    """Physical parameters of the flow."""
    nu: float; mu0: float; sigma0: float; C: float


@functools.partial(jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=['dt', 'steps'])
@dataclass(frozen=True)
class Time:
    """Time integration parameters."""
    dt: float; steps: int

def make_grid(N: int, L: float = 1.0) -> Grid:
    """Construct a uniform 2D periodic grid on [0, L]^2."""
    dx = L / N
    x = jnp.linspace(0.0, L, N, endpoint=False)
    X, Y = jnp.meshgrid(x, x, indexing="ij")
    return Grid(N, L, dx, X, Y)


def load_config(path: Optional[str] = None) -> SimConfig:
    """
    Load a SimConfig from a JSON file, falling back to defaults for any
    missing fields.

    The JSON file may contain any subset of SimConfig fields, e.g.:
        {
            "N": 64,
            "nu": 0.005,
            "t": 0.5,
            "advection_scheme": "upwind"
        }

    Fields not present in the file keep their SimConfig defaults.
    If path is None or the file does not exist, the full default config
    is returned and a warning is logged.

    Args:
        path: Path to a JSON config file, or None.

    Returns:
        A SimConfig instance.
    """
    defaults = {f.name: getattr(SimConfig(), f.name)
                for f in SimConfig.__dataclass_fields__.values()
                if f.name != 'theta0'}

    if path is None:
        logger.info("No config file specified — using default SimConfig.")
        logger.info(f"Using default config: N={CONFIG.N}, nu={CONFIG.nu}, scheme={CONFIG.advection_scheme}, steps = {CONFIG.steps}, dt={CONFIG.dt:.6e}")
        return SimConfig()

    if not os.path.exists(path):
        logger.warning(f"Config file not found: '{path}' — using default SimConfig.")
        return SimConfig()

    with open(path, 'r') as f:
        user = json.load(f)

    unknown = set(user) - set(defaults) - {'theta0'}
    if unknown:
        logger.warning(f"Ignoring unrecognised config fields: {unknown}")

    merged = {**defaults, **{k: v for k, v in user.items() if k in defaults}}

    theta0 = None
    if 'theta0' in user:
        theta0 = jnp.array(user['theta0'])

    cfg = SimConfig(**merged, theta0=theta0)
    logger.info(f"Loaded config from '{path}': N={cfg.N}, nu={cfg.nu}, "
                f"scheme={cfg.advection_scheme}, time={cfg.t}, steps={cfg.steps}, dt={cfg.dt:.6e}")
    return cfg


# Create default configuration (overridden in main() if a config file is passed)
CONFIG = SimConfig()

# ==========================
# 1. Differential Operators
#
# Encapsulate the spatial operators ∇, ∇², ∇· on the periodic domain.
# The two implementations: single-core (roll-based) and distributed
# (halo-exchange-based), share the same interface so the projection
# step is agnostic to execution mode.
# ==========================
class DifferentialOps:
    """
    Differential operators on a 2D periodic uniform grid.

    Defines the interface for ∇ (grad), ∇² (laplacian), ∇· (divergence),
    and direction-aware first-order upwind differencing.
    Subclasses implement the stencil under different parallelism models.
    """

    def __init__(self, dx: float):
        self.dx = dx

    def grad(self, field: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Central-difference gradient: returns (∂/∂x, ∂/∂y)."""
        raise NotImplementedError

    def laplacian(self, field: jax.Array) -> jax.Array:
        """5-point Laplacian ∇²."""
        raise NotImplementedError

    def divergence(self, u: jax.Array) -> jax.Array:
        """Divergence of a 2-component vector field (2, N, N)."""
        gx, _ = self.grad(u[0])
        _, gy = self.grad(u[1])
        return gx + gy

    def upwind_advect(self, field: jax.Array, ux: jax.Array, uy: jax.Array) -> jax.Array:
        """First-order upwind advection: -(u·∇)φ."""
        raise NotImplementedError

    def divergence_vec(self, u: jax.Array) -> jax.Array:
        """Divergence of a 2-component vector field (2, N_x, N_y). Default delegates to divergence."""
        return self.divergence(u)

    def laplacian_and_advect_vec(
        self, u: jax.Array, advection: 'AdvectionScheme'
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute laplacian and advection for a 2-component velocity field.

        Returns (lap_u, adv_u) each of shape (2, N_x, N_y).
        Default implementation makes two separate operator calls; subclasses may fuse them.
        """
        ux, uy = u[0], u[1]
        lap = jnp.stack([self.laplacian(ux), self.laplacian(uy)], axis=0)
        adv = jnp.stack(
            [advection.advect(ux, ux, uy, self), advection.advect(uy, ux, uy, self)],
            axis=0,
        )
        return lap, adv


class PeriodicDifferentialOps(DifferentialOps):
    """
    Single-core differential operators using jnp.roll for periodic wrapping.
    All stencils are O(Δx²) central differences except upwind (O(Δx)).
    """

    def grad(self, field: jax.Array) -> Tuple[jax.Array, jax.Array]:
        dx = self.dx
        gx = (jnp.roll(field, -1, axis=0) - jnp.roll(field, 1, axis=0)) / (2 * dx)
        gy = (jnp.roll(field, -1, axis=1) - jnp.roll(field, 1, axis=1)) / (2 * dx)
        return gx, gy

    def laplacian(self, field: jax.Array) -> jax.Array:
        dx = self.dx
        left = jnp.roll(field,  1, axis=0)
        right = jnp.roll(field, -1, axis=0)
        up = jnp.roll(field, -1, axis=1)
        down = jnp.roll(field,  1, axis=1)
        return (left + right + up + down - 4 * field) / (dx ** 2)

    def upwind_advect(self, field: jax.Array, ux: jax.Array, uy: jax.Array) -> jax.Array:
        dx = self.dx
        f_left = jnp.roll(field,  1, axis=0)
        f_right = jnp.roll(field, -1, axis=0)
        dfx = jnp.where(ux >= 0, field - f_left, f_right - field) / dx
        f_down = jnp.roll(field,  1, axis=1)
        f_up = jnp.roll(field, -1, axis=1)
        dfy = jnp.where(uy >= 0, field - f_down, f_up - field) / dx
        return -(ux * dfx + uy * dfy)


class DistributedDifferentialOps(DifferentialOps):
    """
    Distributed differential operators using halo exchange via jax.lax.ppermute.

    Each device owns a slab of N_local = N // device_number rows along x.
    Ghost cells from neighbouring devices are exchanged before stencil evaluation,
    while the y-direction remains fully local (periodic via roll).
    """

    def __init__(self, dx: float, device_number: int, axis_name: str = 'x'):
        super().__init__(dx)
        self.device_number = device_number
        self.axis_name = axis_name

    def _halo_pad_x(self, local_field: jax.Array, x_axis: int = 0) -> jax.Array:
        """
        Exchange one-cell ghost rows with left/right neighbours via ppermute,
        then concatenate: [ghost_left | local | ghost_right].
        """
        n = self.device_number
        name = self.axis_name
        first = [slice(None)] * local_field.ndim
        first[x_axis] = slice(0, 1)
        last = [slice(None)] * local_field.ndim
        last[x_axis] = slice(-1, None)
        first = tuple(first)
        last = tuple(last)
        left_halo = jax.lax.ppermute(
            local_field[last], axis_name=name,
            perm=[(i, (i + 1) % n) for i in range(n)]
        )
        right_halo = jax.lax.ppermute(
            local_field[first], axis_name=name,
            perm=[(i, (i - 1) % n) for i in range(n)]
        )
        return jnp.concatenate([left_halo, local_field, right_halo], axis=x_axis)

    def grad(self, field: jax.Array) -> Tuple[jax.Array, jax.Array]:
        padded = self._halo_pad_x(field)
        gx = (padded[2:] - padded[:-2]) / (2.0 * self.dx)
        gy = (jnp.roll(field, -1, axis=1) - jnp.roll(field, 1, axis=1)) / (2.0 * self.dx)
        return gx, gy

    def laplacian(self, field: jax.Array) -> jax.Array:
        padded = self._halo_pad_x(field)
        d2x = (padded[:-2] + padded[2:] - 2 * padded[1:-1]) / (self.dx ** 2)
        up = jnp.roll(field, -1, axis=1)
        down = jnp.roll(field,  1, axis=1)
        d2y = (up + down - 2 * field) / (self.dx ** 2)
        return d2x + d2y

    def upwind_advect(self, field: jax.Array, ux: jax.Array, uy: jax.Array) -> jax.Array:
        padded = self._halo_pad_x(field)
        f_left = padded[:-2]
        f_right = padded[2:]
        dfx = jnp.where(ux >= 0, field - f_left, f_right - field) / self.dx
        f_down = jnp.roll(field,  1, axis=1)
        f_up = jnp.roll(field, -1, axis=1)
        dfy = jnp.where(uy >= 0, field - f_down, f_up - field) / self.dx
        return -(ux * dfx + uy * dfy)

    def grad_vec(self, field_vec: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Vectorized gradient for a stack of scalar fields.
        field_vec shape: (C, N_local, N), returns gx, gy with same shape.
        """
        padded = self._halo_pad_x(field_vec, x_axis=1)
        gx = (padded[:, 2:, :] - padded[:, :-2, :]) / (2.0 * self.dx)
        gy = (jnp.roll(field_vec, -1, axis=2) - jnp.roll(field_vec, 1, axis=2)) / (2.0 * self.dx)
        return gx, gy

    def laplacian_vec(self, field_vec: jax.Array) -> jax.Array:
        """
        Vectorized 5-point Laplacian for stacked scalar fields.
        field_vec shape: (C, N_local, N), returns same shape.
        """
        padded = self._halo_pad_x(field_vec, x_axis=1)
        d2x = (padded[:, :-2, :] + padded[:, 2:, :] - 2 * padded[:, 1:-1, :]) / (self.dx ** 2)
        up = jnp.roll(field_vec, -1, axis=2)
        down = jnp.roll(field_vec, 1, axis=2)
        d2y = (up + down - 2 * field_vec) / (self.dx ** 2)
        return d2x + d2y

    def upwind_advect_vec(self, field_vec: jax.Array, ux: jax.Array, uy: jax.Array) -> jax.Array:
        """
        Vectorized first-order upwind advection for stacked fields.
        field_vec shape: (C, N_local, N), ux/uy shape: (N_local, N).
        """
        padded = self._halo_pad_x(field_vec, x_axis=1)
        f_left = padded[:, :-2, :]
        f_right = padded[:, 2:, :]
        dfx = jnp.where(ux[None, :, :] >= 0, field_vec - f_left, f_right - field_vec) / self.dx
        f_down = jnp.roll(field_vec, 1, axis=2)
        f_up = jnp.roll(field_vec, -1, axis=2)
        dfy = jnp.where(uy[None, :, :] >= 0, field_vec - f_down, f_up - field_vec) / self.dx
        return -(ux[None, :, :] * dfx + uy[None, :, :] * dfy)

    def laplacian_and_grad_vec(
        self, field_vec: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Laplacian and gradient in a single halo exchange.

        Returns (lap, gx, gy) each of shape (C, N_local, N).
        Saves one halo exchange compared to calling laplacian_vec + grad_vec separately.
        """
        padded = self._halo_pad_x(field_vec, x_axis=1)
        f_left  = padded[:, :-2, :]
        f_right = padded[:, 2:,  :]
        f_mid   = padded[:, 1:-1, :]
        d2x = (f_left + f_right - 2 * f_mid) / (self.dx ** 2)
        gx  = (f_right - f_left) / (2.0 * self.dx)
        up   = jnp.roll(field_vec, -1, axis=2)
        down = jnp.roll(field_vec,  1, axis=2)
        d2y = (up + down - 2 * field_vec) / (self.dx ** 2)
        gy  = (up - down) / (2.0 * self.dx)
        return d2x + d2y, gx, gy

    def laplacian_and_upwind_advect_vec(
        self, field_vec: jax.Array, ux: jax.Array, uy: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Laplacian and upwind advection in a single halo exchange.

        Returns (lap, adv) each of shape (C, N_local, N).
        Saves one halo exchange compared to calling laplacian_vec + upwind_advect_vec separately.
        """
        padded = self._halo_pad_x(field_vec, x_axis=1)
        f_left  = padded[:, :-2, :]
        f_right = padded[:, 2:,  :]
        f_mid   = padded[:, 1:-1, :]
        d2x = (f_left + f_right - 2 * f_mid) / (self.dx ** 2)
        dfx = jnp.where(ux[None, :, :] >= 0, f_mid - f_left, f_right - f_mid) / self.dx
        up   = jnp.roll(field_vec, -1, axis=2)
        down = jnp.roll(field_vec,  1, axis=2)
        d2y = (up + down - 2 * field_vec) / (self.dx ** 2)
        dfy = jnp.where(uy[None, :, :] >= 0, field_vec - down, up - field_vec) / self.dx
        return d2x + d2y, -(ux[None, :, :] * dfx + uy[None, :, :] * dfy)

    def laplacian_and_advect_vec(
        self, u: jax.Array, advection: 'AdvectionScheme'
    ) -> Tuple[jax.Array, jax.Array]:
        """Fused laplacian + advection using a single halo exchange.

        Returns (lap_u, adv_u) each of shape (2, N_local, N).
        Dispatches to the appropriate fused kernel based on advection scheme type.
        """
        ux, uy = u[0], u[1]
        if isinstance(advection, CentralAdvection):
            lap_u, gx_u, gy_u = self.laplacian_and_grad_vec(u)
            adv_u = jnp.stack([
                -(ux * gx_u[0] + uy * gy_u[0]),
                -(ux * gx_u[1] + uy * gy_u[1]),
            ], axis=0)
        elif isinstance(advection, UpwindAdvection):
            lap_u, adv_u = self.laplacian_and_upwind_advect_vec(u, ux, uy)
        else:
            lap_u = self.laplacian_vec(u)
            adv_u = jnp.stack(
                [advection.advect(ux, ux, uy, self), advection.advect(uy, ux, uy, self)],
                axis=0,
            )
        return lap_u, adv_u

    def divergence_vec(self, field_vec: jax.Array) -> jax.Array:
        """Divergence of a (2, N_local, N) velocity field.

        Only pads field_vec[0] (the x-component) for the x-gradient.
        Transfers half the data of grad_vec, since field_vec[1]'s y-gradient is
        purely local (jnp.roll).
        """
        padded_x = self._halo_pad_x(field_vec[0])          # (N_local, N) → (N_local+2, N)
        gx = (padded_x[2:] - padded_x[:-2]) / (2.0 * self.dx)
        gy = (jnp.roll(field_vec[1], -1, axis=1) - jnp.roll(field_vec[1], 1, axis=1)) / (2.0 * self.dx)
        return gx + gy


# ==========================
# 2. Advection Schemes
#
# Strategy pattern: the ChorinProjector holds one AdvectionScheme instance
# and delegates -(u·∇)u to it. Swapping schemes requires no changes to the
# projection logic.
# ==========================
class AdvectionScheme:
    """Interface for advection discretizations of the nonlinear term -(u·∇)u."""

    def advect(self, field: jax.Array, ux: jax.Array, uy: jax.Array,
               ops: DifferentialOps) -> jax.Array:
        """
        Compute advective contribution for one velocity component.

        Args:
            field: scalar field being advected (ux or uy), shape (N_x, N_y)
            ux, uy: velocity components for upwinding direction
            ops: differential operator instance (provides stencil access)
        Returns:
            -(u·∇)field, shape (N_x, N_y)
        """
        raise NotImplementedError


class CentralAdvection(AdvectionScheme):
    """
    Second-order central-difference advection: -(u·∇)φ = -(ux·∂φ/∂x + uy·∂φ/∂y).
    More accurate but can produce dispersive oscillations at low viscosity.
    """

    def advect(self, field: jax.Array, ux: jax.Array, uy: jax.Array,
               ops: DifferentialOps) -> jax.Array:
        fx, fy = ops.grad(field)
        return -(ux * fx + uy * fy)


class UpwindAdvection(AdvectionScheme):
    """
    First-order upwind advection: differencing side selected by sign of velocity.
    Monotone and stable at high Reynolds number, at the cost of numerical diffusion.
    """

    def advect(self, field: jax.Array, ux: jax.Array, uy: jax.Array,
               ops: DifferentialOps) -> jax.Array:
        return ops.upwind_advect(field, ux, uy)


def make_advection_scheme(name: str) -> AdvectionScheme:
    """Factory: return an AdvectionScheme by name ('central' or 'upwind')."""
    if name == "central":
        return CentralAdvection()
    elif name == "upwind":
        return UpwindAdvection()
    raise ValueError(f"Unknown advection scheme: '{name}'. Choose 'central' or 'upwind'.")


# ==========================
# 3. Poisson Solver (Pressure Projection)
#
# Solves -∇²p = rhs spectrally using FFT.
# The distributed variant adds a pair of all-to-all transposes to perform
# the 2D FFT across a domain decomposed in x.
# ==========================
class SpectralPoissonSolver:
    """
    Spectral solver for the pressure Poisson equation: -∇²p = rhs.

    The solution is exact up to floating-point precision on a periodic domain.
    The zero-wavenumber mode is pinned to zero to fix the gauge (mean pressure).

    Precomputes the eigenvalue array 1/|k|² at construction time.
    """

    def __init__(self, Nx: int, Ny: int, dx: float):
        kx = jnp.fft.fftfreq(Nx, d=dx) * 2 * jnp.pi
        ky = jnp.fft.fftfreq(Ny, d=dx) * 2 * jnp.pi
        KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
        self._inv_k2 = jnp.where((KX**2 + KY**2) == 0, 0.0, 1.0 / (KX**2 + KY**2))

    def __call__(self, rhs: jax.Array) -> jax.Array:
        """Solve -∇²p = rhs and return p."""
        p_hat = -jnp.fft.fft2(rhs) * self._inv_k2
        p_hat = p_hat.at[0, 0].set(0.0)
        return jnp.fft.ifft2(p_hat).real


class DistributedSpectralPoissonSolver(SpectralPoissonSolver):
    """
    Distributed spectral Poisson solver for domain-decomposed execution.

    Each device holds a slab of shape (N_local, N) along x. The 2D FFT
    requires a global transpose (all-to-all) to complete the x-direction
    transform, followed by an inverse transpose after the solve.

    Sequence: FFT_y → all-to-all → FFT_x → solve in k-space → iFFT_x → all-to-all⁻¹ → iFFT_y
    """

    def __init__(self, Nx: int, Ny: int, dx: float, device_number: int):
        # Do not call super().__init__() — inv_k2 is computed per-shard at call time
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.device_number = device_number
        if Ny % device_number != 0:
            raise ValueError(f"Ny={Ny} must be divisible by device_number={device_number}")
        # Store global wavenumber axes; each shard selects its own ky slice at call time.
        # Memory: O(N) per axis, not O(N²) — the per-shard inv_k2 is built inside __call__.
        self._kx_global = jnp.fft.fftfreq(Nx, d=dx) * 2 * jnp.pi
        self._ky_global = jnp.fft.fftfreq(Ny, d=dx) * 2 * jnp.pi

    def _transpose(self, x: jax.Array) -> jax.Array:
        """all-to-all: (N_local, N) → (N, N_local)."""
        n_local, n_global = x.shape
        x_r = x.reshape(n_local, self.device_number, -1)
        x_t = jax.lax.all_to_all(x_r, axis_name='x', split_axis=1, concat_axis=0)
        return x_t.reshape(n_local * self.device_number, -1)

    def _transpose_inv(self, x: jax.Array) -> jax.Array:
        """Inverse all-to-all: (N, N_local) → (N_local, N).

        Equivalent to _transpose(x.T).T — reuses split_axis=1, concat_axis=0
        which gives block (not interleaved) ordering.
        """
        n_global, n_local = x.shape
        xt = x.T # (N_local, N)
        x_r = xt.reshape(n_local, self.device_number, -1) # (N_local, P, N_local)
        x_t = jax.lax.all_to_all(x_r, axis_name='x', split_axis=1, concat_axis=0)
        return x_t.reshape(n_global, -1).T # (N_local, N)

    def __call__(self, rhs_local: jax.Array) -> jax.Array:
        """Distributed solve -∇²p = rhs on the local slab."""
        hat_y = jnp.fft.fft(rhs_local, axis=1) # FFT along y (local)
        hat_t = self._transpose(hat_y) # all-to-all → (N, Ny_local)
        hat_xy = jnp.fft.fft(hat_t, axis=0) # FFT along x (now global)

        # Build per-shard inv_k2 from the device's own ky slice — O(N²/P) per device.
        my_id = jax.lax.axis_index('x')
        ny_local = self.Ny // self.device_number
        ky_local = jax.lax.dynamic_slice(self._ky_global, (my_id * ny_local,), (ny_local,))
        KX, KY = jnp.meshgrid(self._kx_global, ky_local, indexing="ij")
        inv_k2 = jnp.where((KX ** 2 + KY ** 2) == 0, 0.0, 1.0 / (KX ** 2 + KY ** 2))

        p_hat = -hat_xy * inv_k2
        # Gauge fix: pin mean pressure to zero (only on device 0, mode (0,0))
        p_hat = jax.lax.cond(
            my_id == 0,
            lambda x: x.at[0, 0].set(0.0),
            lambda x: x,
            p_hat,
        )

        p_t = jnp.fft.ifft(p_hat, axis=0) # iFFT along x
        p_y = self._transpose_inv(p_t) # all-to-all⁻¹ → (N_local, N)
        return jnp.fft.ifft(p_y, axis=1).real # iFFT along y


# ==========================
# 4. Forcing Field
#
# Encapsulates the parameterised external body force f(x, t; θ).
# θ = [A, B] modulates the Gaussian centre and width over time.
# ==========================
class GaussianForcingField:
    """
    Time-varying Gaussian body force in the x-direction:
        f(x, t; θ) = C · exp(-d(x, μ(t))² / 2σ(t)²) · ê_x

    where:
        μ(t) = μ₀ + sin(2π·A·t)      (travelling centre)
        σ(t) = σ₀ + cos²(2π·B·t)     (pulsating width)
        d(x, μ) = wrapped distance on [0, L] (periodic)
    """

    def __init__(self, grid: Grid, phys: Phys):
        self.grid = grid
        self.phys = phys

    def _x_component(self, theta: jax.Array, t: float, x: jax.Array) -> jax.Array:
        """Evaluate the scalar Gaussian forcing profile at x-coordinates."""
        A, B = theta[0], theta[1]
        mu_t = self.phys.mu0 + jnp.sin(2.0 * jnp.pi * A * t)
        sig_t = self.phys.sigma0 + jnp.cos(2.0 * jnp.pi * B * t) ** 2
        d = (x - mu_t) - jnp.round(x - mu_t)
        return self.phys.C * jnp.exp(-(d * d) / (2.0 * sig_t * sig_t))

    def local_field(self, theta: jax.Array, t: float, row_start: int, rows: int) -> jax.Array:
        """Return the forcing field restricted to a contiguous x-slab."""
        x_idx_local = row_start + jnp.arange(rows)
        x_local = x_idx_local * self.grid.dx
        fx_local = self._x_component(theta, t, x_local[:, None])
        fx_local = jnp.broadcast_to(fx_local, (rows, self.grid.N))
        return jnp.stack([fx_local, jnp.zeros_like(fx_local)], axis=0)

    def __call__(self, theta: jax.Array, t: float) -> jax.Array:
        """Return forcing field of shape (2, N, N) at time t for parameters θ."""
        fx = self._x_component(theta, t, self.grid.X)
        return jnp.stack([fx, jnp.zeros_like(fx)], axis=0)


# ==========================
# 5. Chorin Projector
#
# The core numerical scheme: one explicit Euler + pressure-projection step
# (Chorin splitting) for the incompressible Navier-Stokes equations.
#
# Given u^n:
#   1. Advection-diffusion:  u* = u^n + dt·(ν∇²u^n − (u^n·∇)u^n + f^n)
#   2. Pressure solve:       −∇²p = (∇·u*)/dt
#   3. Velocity projection:  u^{n+1} = u* − dt·∇p
# ==========================
class ChorinProjector:
    """
    One-step Chorin projection method for incompressible NS.

    Holds references to the differential operators, Poisson solver, and
    advection scheme — all of which are injected at construction, making
    this class independent of parallelism strategy.
    """

    def __init__(self, ops: DifferentialOps, poisson: SpectralPoissonSolver,
                 advection: AdvectionScheme, nu: float, dt: float):
        self.ops = ops
        self.poisson = poisson
        self.advection = advection
        self.nu = nu
        self.dt = dt

    def step(self, u: jax.Array, f: jax.Array) -> jax.Array:
        """
        Advance velocity field u by one time step given body force f.

        Args:
            u: velocity field, shape (2, N_x, N_y)
            f: body force,     shape (2, N_x, N_y)
        Returns:
            u^{n+1}, shape (2, N_x, N_y)
        """
        ux, uy = u[0], u[1]
        ops, dt, nu = self.ops, self.dt, self.nu

        # --- Step 1: Advection-diffusion (explicit Euler) ---
        lap_u, adv_u = ops.laplacian_and_advect_vec(u, self.advection)

        ux_star = ux + dt * (nu * lap_u[0] + adv_u[0] + f[0])
        uy_star = uy + dt * (nu * lap_u[1] + adv_u[1] + f[1])
        u_star = jnp.stack([ux_star, uy_star], axis=0)

        # --- Step 2: Pressure solve −∇²p = (∇·u*)/dt ---
        div_star = ops.divergence_vec(u_star)
        p = self.poisson(div_star / dt)

        # --- Step 3: Divergence-free projection ---
        px, py = ops.grad(p)
        return jnp.stack([ux_star - dt * px, uy_star - dt * py], axis=0)


# ==========================
# 6. NS Solvers
#
# Top-level solver objects that drive the rollout loop, accumulate the
# objective J = ∫∫ u·f dx dt, and expose JIT-compiled objective and
# gradient callables for optimisation.
# ==========================
class SingleCoreNSSolver:
    """
    Single-device NS solver.

    Wraps a ChorinProjector with PeriodicDifferentialOps and a single-core
    SpectralPoissonSolver. Exposes rollout, objective, and gradient.
    """

    def __init__(self, grid: Grid, phys: Phys, time_cfg: Time,
                 advection_scheme: str = "central"):
        self.grid = grid
        self.time = time_cfg
        ops = PeriodicDifferentialOps(grid.dx)
        poisson = SpectralPoissonSolver(grid.N, grid.N, grid.dx)
        advection = make_advection_scheme(advection_scheme)
        self.forcing = GaussianForcingField(grid, phys)
        self.projector = ChorinProjector(ops, poisson, advection, phys.nu, time_cfg.dt)
        self.jit_objective = jax.jit(self.objective)
        self.jit_gradient = jax.jit(jax.grad(self.objective))

    def step(self, u: jax.Array, theta: jax.Array, t: float) -> jax.Array:
        """Single time step."""
        f = self.forcing(theta, t)
        return self.projector.step(u, f)

    def rollout(self, u0: jax.Array, theta: jax.Array) -> Tuple[float, jax.Array]:
        """
        Integrate from u0 for self.time.steps steps, accumulating objective:
            J = Σ_t  dt · ∫∫ u(t) · f(t; θ) dx
        Returns (J, u_final).
        """
        dt, dx = self.time.dt, self.grid.dx
        forcing = self.forcing
        projector = self.projector

        def body(carry, i):
            u, J = carry
            t = i * dt
            f = forcing(theta, t)
            J += dt * jnp.sum(u * f) * (dx ** 2)
            return (projector.step(u, f), J), None

        (u_final, J), _ = jax.lax.scan(body, (u0, 0.0), jnp.arange(self.time.steps))
        return J, u_final

    def objective(self, theta: jax.Array) -> float:
        """Compute J(θ) from zero initial condition."""
        u0 = jnp.zeros((2, self.grid.N, self.grid.N))
        J, _ = self.rollout(u0, theta)
        return J


class DistributedNSSolver:
    """
    Multi-device NS solver using JAX shard_map (SPMD).

    The domain is decomposed into equal slabs along x, one per device.
    Halo exchange handles inter-device stencil communication;
    all-to-all transposes enable the distributed spectral Poisson solve;
    lax.psum reduces the per-slab objective contributions to a scalar.
    """

    def __init__(self, grid: Grid, phys: Phys, time_cfg: Time,
                 mesh: jax.sharding.Mesh,
                 device_number: Optional[int] = None,
                 advection_scheme: str = "central"):
        if device_number is None:
            device_number = len(mesh.devices.flat)
        if grid.N % device_number != 0:
            raise ValueError(
                f"Grid resolution N={grid.N} must be divisible by device_number={device_number}"
            )
        self.grid = grid
        self.time = time_cfg
        self.mesh = mesh
        self.device_number = device_number

        ops = DistributedDifferentialOps(grid.dx, device_number)
        poisson = DistributedSpectralPoissonSolver(grid.N, grid.N, grid.dx, device_number)
        advection = make_advection_scheme(advection_scheme)
        self.forcing = GaussianForcingField(grid, phys)
        self.projector = ChorinProjector(ops, poisson, advection, phys.nu, time_cfg.dt)

        self._step_sharded = self._build_sharded_step()
        self._rollout_sharded = self._build_sharded_rollout()
        self._u0_sharding = NamedSharding(mesh, P(None, 'x', None))
        self.jit_objective = jax.jit(self.objective)
        self.jit_gradient = jax.jit(jax.grad(self.objective))

    # ------------------------------------------------------------------
    # Private: build shard_map-wrapped kernels at construction time
    # ------------------------------------------------------------------
    def _build_sharded_step(self):
        """Wrap one projection step in shard_map."""
        grid = self.grid
        forcing = self.forcing
        projector = self.projector
        device_number = self.device_number

        def physics_kernel(u_local: jax.Array, theta: jax.Array, t: float) -> jax.Array:
            my_id = jax.lax.axis_index('x')
            rows = grid.N // device_number
            f_local = forcing.local_field(theta, t, row_start=my_id * rows, rows=rows)
            return projector.step(u_local, f_local)

        return shard_map(
            physics_kernel,
            mesh=self.mesh,
            in_specs=(P(None, 'x', None), P(None), P()),
            out_specs=P(None, 'x', None),
            check_rep=False,
        )

    def _build_sharded_rollout(self):
        """Wrap distributed rollout with local accumulation and one final global reduction."""
        grid = self.grid
        forcing = self.forcing
        device_number = self.device_number
        dt = self.time.dt
        mesh = self.mesh
        projector = self.projector

        def rollout_local_kernel(u0_local: jax.Array, theta: jax.Array) -> Tuple[jax.Array, jax.Array]:
            """
            Per-device rollout:
              - compute local forcing once per step
              - accumulate local objective without cross-device sync
            """
            my_id = jax.lax.axis_index('x')
            rows = grid.N // device_number

            def body(carry, i):
                u_local, J_local = carry
                t = i * dt
                f_local = forcing.local_field(theta, t, row_start=my_id * rows, rows=rows)
                # Match single-core timing: accumulate using u(t), then step to u(t+dt).
                J_local += dt * jnp.sum(u_local * f_local) * (grid.dx ** 2)
                u_next_local = projector.step(u_local, f_local)
                return (u_next_local, J_local), None

            (u_final_local, J_local), _ = jax.lax.scan(
                body, (u0_local, 0.0), jnp.arange(self.time.steps)
            )
            # Add a dummy axis so out_specs can shard this across mesh axis 'x'.
            return u_final_local, J_local[None]

        rollout_local_sharded = jax.jit(shard_map(
            rollout_local_kernel,
            mesh=mesh,
            in_specs=(P(None, 'x', None), P(None)),
            out_specs=(P(None, 'x', None), P('x')),
            check_rep=False,
        ))

        def rollout_fn(u0: jax.Array, theta: jax.Array) -> Tuple[float, jax.Array]:
            u_final, J_local_vec = rollout_local_sharded(u0, theta)
            J = jnp.sum(J_local_vec)
            return J, u_final

        return jax.jit(rollout_fn)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def step(self, u: jax.Array, theta: jax.Array, t: float) -> jax.Array:
        """Single distributed time step (u must be sharded)."""
        return self._step_sharded(u, theta, t)

    def rollout(self, u0: jax.Array, theta: jax.Array) -> Tuple[float, jax.Array]:
        """Distributed rollout. u0 must be pre-sharded via jax.device_put."""
        return self._rollout_sharded(u0, theta)

    def objective(self, theta: jax.Array) -> float:
        """Compute J(θ) from zero initial condition (sharding handled internally)."""
        u0 = jax.device_put(jnp.zeros((2, self.grid.N, self.grid.N)), self._u0_sharding)
        J, _ = self.rollout(u0, theta)
        return J


# ==========================
# 7. Validation Helper
# ==========================
def validate_solvers(single: SingleCoreNSSolver, distributed: DistributedNSSolver,
                     mesh: jax.sharding.Mesh, theta: jax.Array, tol: float = 1e-6):
    """
    Run both solvers from zero initial condition and compare final velocity fields.
    Logs relative error ||u_single - u_dist|| / ||u_single||.
    """
    N = single.grid.N
    U_SHARDING = NamedSharding(mesh, P(None, 'x', None))

    logger.info("Validating full rollout (single-core vs distributed)...")
    t0 = time.perf_counter()
    _, u_single = single.rollout(jnp.zeros((2, N, N)), theta)
    jax.block_until_ready(u_single)
    t_single = time.perf_counter() - t0
    logger.info(f"Total time used for single-core solver: {t_single:.3f}s")
    u0_dist = jax.device_put(jnp.zeros((2, N, N)), U_SHARDING)
    t0 = time.perf_counter()
    _, u_dist = distributed.rollout(u0_dist, theta)
    jax.block_until_ready(u_dist)
    t_dist = time.perf_counter() - t0
    logger.info(f"Total time used for distributed solver: {t_dist:.3f}s")

    u_dist_host = jax.device_get(u_dist)
    err = jnp.linalg.norm(u_single - u_dist_host) / (jnp.linalg.norm(u_single) + 1e-12)
    logger.info(f"Relative state error ||u_single - u_dist|| / ||u_single||: {err:.3e}")

    if err < tol:
        logger.info("SUCCESS: Single-core and distributed states match.")
    else:
        logger.warning(f"FAIL: State mismatch exceeds tolerance {tol:.1e}")

    return err, u_single, u_dist_host, t_single, t_dist


# ==========================
# 8. Main
# ==========================
if __name__ == "__main__":
    from runner import main
    main()
