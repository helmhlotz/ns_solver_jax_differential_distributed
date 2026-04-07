"""research studies module for CFD solver analysis.

This module contains independent study classes for analyzing CFD solver behavior:
- ViscosityStudy: Systematic study across viscosity regimes
- GradientBenchmark: JAX AD vs finite-difference gradient comparison
- SchemeComparisonStudy: Central vs upwind advection scheme comparison
- ParameterOptimizationStudy: Optimizer comparison with Optax

Can be run standalone or imported from other modules.
"""

from typing import Callable, Tuple
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import optax

# Import from the optimized solver module
from NS_solver import (
    CONFIG,
    Grid,
    Phys,
    SimConfig,
    Time,
    make_grid,
    load_config,
    SingleCoreNSSolver,
    GaussianForcingField,
    logger,
)


def _robust_color_limits(values, p_low: float = 2.0, p_high: float = 98.0) -> Tuple[float, float]:
    """Compute robust color limits with safe fallback for near-constant fields."""
    arr = np.asarray(values).ravel()
    vmin = float(np.percentile(arr, p_low))
    vmax = float(np.percentile(arr, p_high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        center = float(np.mean(arr))
        eps = max(1e-12, abs(center) * 1e-12)
        return center - eps, center + eps
    return vmin, vmax

# ==========================
# 1. Viscosity Regime Study
# ==========================
class ViscosityStudy:
    """Systematic study of CFD solver behavior across viscosity regimes."""
    def __init__(
        self,
        viscosities: jax.Array = None,
        grid: Grid = None,
        phys_template: Phys = None,
        time_template: Time = None,
        theta: jax.Array = None,
        config: SimConfig = None
    ):
        if viscosities is None:
            self.viscosities = jnp.array([0.001, 0.005, 0.01, 0.05, 0.1])
        else:
            self.viscosities = viscosities
        
        self.grid = grid if grid is not None else make_grid(128, 1.0)
        self.phys_template = phys_template
        self.time_template = time_template
        self.theta = theta if theta is not None else jnp.array([1.0, 0.5])
        self.config = config if config is not None else CONFIG
        self.results = {}
        
        logger.info(f"ViscosityStudy initialized with {len(self.viscosities)} viscosity values")
    
    def _compute_kinetic_energy(self, u: jax.Array) -> float:
        return float(0.5 * jnp.sum(u[0]**2 + u[1]**2) * (self.grid.dx**2))
    
    def _compute_vorticity(self, u: jax.Array, dx: float) -> jax.Array:
        """vorticity field ω = ∂v/∂x - ∂u/∂y."""
        uy = u[1]
        ux = u[0]
        duy_dx = (jnp.roll(uy, -1, axis=0) - jnp.roll(uy, 1, axis=0)) / (2*dx)
        dux_dy = (jnp.roll(ux, -1, axis=1) - jnp.roll(ux, 1, axis=1)) / (2*dx)
        vorticity = duy_dx - dux_dy
        
        # Debug: Log vorticity statistics
        logger.debug(f"Vorticity stats - min: {jnp.min(jnp.abs(vorticity)):.3e}, "
                    f"max: {jnp.max(jnp.abs(vorticity)):.3e}, "
                    f"mean: {jnp.mean(jnp.abs(vorticity)):.3e}")
        
        return vorticity
    
    def _compute_velocity_stats(self, u: jax.Array) -> dict:
        """velocity field statistics."""
        speed = jnp.sqrt(u[0]**2 + u[1]**2)
        return {
            'speed_max': float(jnp.max(speed)),
            'speed_mean': float(jnp.mean(speed)),
            'speed_std': float(jnp.std(speed)),
            'ux_max': float(jnp.max(jnp.abs(u[0]))),
            'uy_max': float(jnp.max(jnp.abs(u[1]))),
        }
    
    def run_sweep(self) -> dict:
        results = {}

        for nu in self.viscosities:
            logger.info(f"Running simulation for nu={nu:.4f}")

            nu_config = SimConfig(
                N=self.config.N,
                L=self.config.L,
                nu=float(nu),
                mu0=self.config.mu0,
                sigma0=self.config.sigma0,
                C_force=self.config.C_force,
                cfl=self.config.cfl,
                v_max=self.config.v_max,
                t=self.config.t,
                theta0=self.theta,
                advection_scheme=self.config.advection_scheme,
                obj_tol=self.config.obj_tol
            )

            grid     = make_grid(nu_config.N, nu_config.L)
            phys     = Phys(nu=nu_config.nu, mu0=nu_config.mu0,
                            sigma0=nu_config.sigma0, C=nu_config.C_force)
            time_obj = Time(dt=nu_config.dt, steps=nu_config.steps)

            solver = SingleCoreNSSolver(grid, phys, time_obj,
                                        advection_scheme=nu_config.advection_scheme)
            u0 = jnp.zeros((2, nu_config.N, nu_config.N))
            objective_val, u_final = solver.rollout(u0, self.theta)
            
            stats = self._compute_velocity_stats(u_final)
            vorticity = self._compute_vorticity(u_final, grid.dx)
            kinetic_energy = self._compute_kinetic_energy(u_final)
            
            results[float(nu)] = {
                'dt': nu_config.dt,
                'objective': float(objective_val),
                'kinetic_energy': kinetic_energy,
                'velocity_stats': stats,
                'vorticity': vorticity,
                'u_final': u_final
            }
            
            logger.info(f"dt={nu_config.dt:.6e}, obj={objective_val:.6e}, KE={kinetic_energy:.6e}")
        
        self.results = results
        return results
    
    def save_results(self, filename: str = "result/viscosity_study_results.npz"):
        """Save to NPZ"""
        if not self.results:
            logger.warning("No results to save")
            return
        
        data = {}
        for nu, res in self.results.items():
            data[f'nu_{nu:.4f}_objective'] = res['objective']
            data[f'nu_{nu:.4f}_kinetic_energy'] = res['kinetic_energy']
            data[f'nu_{nu:.4f}_dt'] = res['dt']
        
        np.savez(filename, **data)
        logger.info(f"Saved results to {filename}")
    
    def plot_velocity_sweep(self, figsize: Tuple[int, int] = (15, 3)):
        if not self.results:
            logger.warning("No results to plot")
            return
        
        n_nu = len(self.viscosities)
        fig, axes = plt.subplots(1, n_nu, figsize=figsize)
        
        if n_nu == 1:
            axes = [axes]

        speed_values = []
        for nu in sorted(self.results.keys()):
            u_final = self.results[nu]['u_final']
            speed_values.append(np.asarray(jnp.sqrt(u_final[0]**2 + u_final[1]**2)).ravel())
        vmin, vmax = _robust_color_limits(np.concatenate(speed_values))
        
        for idx, nu in enumerate(sorted(self.results.keys())):
            u_final = self.results[nu]['u_final']
            speed = jnp.sqrt(u_final[0]**2 + u_final[1]**2)
            
            im = axes[idx].imshow(
                speed.T,
                origin='lower',
                extent=[0, self.grid.L, 0, self.grid.L],
                cmap='viridis',
                vmin=vmin,
                vmax=vmax,
            )
            axes[idx].set_title(f'nu={nu:.4f}')
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('y')
            plt.colorbar(im, ax=axes[idx], label='Speed')
        
        plt.tight_layout()
        plt.savefig('result/viscosity_study_velocity_sweep.png', dpi=150)
        logger.info("Saved to result/viscosity_study_velocity_sweep.png")
        plt.close()
    
    def plot_vorticity_sweep(self, figsize: Tuple[int, int] = (15, 3)):
        if not self.results:
            logger.warning("No results to plot")
            return
        
        n_nu = len(self.viscosities)
        fig, axes = plt.subplots(1, n_nu, figsize=figsize)
        
        if n_nu == 1:
            axes = [axes]
        
        for idx, nu in enumerate(sorted(self.results.keys())):
            vorticity = self.results[nu]['vorticity']
            
            im = axes[idx].imshow(vorticity.T, origin='lower', extent=[0, self.grid.L, 0, self.grid.L], cmap='RdBu_r')
            axes[idx].set_title(f'ν={nu:.4f}')
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('y')
            plt.colorbar(im, ax=axes[idx], label='Vorticity')
        
        plt.tight_layout()
        plt.savefig('result/viscosity_study_vorticity_sweep.png', dpi=150)
        logger.info("Saved vorticity sweep visualization to result/viscosity_study_vorticity_sweep.png")
        plt.close()
    
    def plot_metrics_comparison(self):
        if not self.results:
            logger.warning("No results to plot")
            return
        
        nus = sorted(self.results.keys())
        objectives = [self.results[nu]['objective'] for nu in nus]
        kinetic_energies = [self.results[nu]['kinetic_energy'] for nu in nus]
        dts = [self.results[nu]['dt'] for nu in nus]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].loglog(nus, objectives, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Viscosity nu')
        axes[0].set_ylabel('Objective J')
        axes[0].set_title('Work Done by Forcing')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].loglog(nus, kinetic_energies, 's-', linewidth=2, markersize=8, color='orange')
        axes[1].set_xlabel('Viscosity nu')
        axes[1].set_ylabel('Kinetic Energy')
        axes[1].set_title('Final Kinetic Energy')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].loglog(nus, dts, '^-', linewidth=2, markersize=8, color='green')
        axes[2].set_xlabel('Viscosity nu')
        axes[2].set_ylabel('Time Step dt')
        axes[2].set_title('Stable Time Step')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('result/viscosity_study_metrics.png', dpi=150)
        logger.info("Saved to result/viscosity_study_metrics.png")
        plt.close()

# ==========================
# 2. Gradient Research Framework
# ==========================
class GradientBenchmark:
    """JAX AD vs finite-difference gradients"""    
    def __init__(
        self,
        objective: Callable[[jax.Array], float],
        grad_jax: Callable[[jax.Array], jax.Array],
        param_vectors: list = None,
        epsilon_range: jax.Array = None,
        jax_timing_runs: int = 5,
        config: SimConfig = None
    ):

        self.objective = objective
        self.grad_jax = grad_jax
        self.config = config or CONFIG
        
        if param_vectors is None:
            self.param_vectors = [
                jnp.array([1.0, 0.5]),
                jnp.array([0.5, 1.0]),
                jnp.array([2.0, 0.2])
            ]
        else:
            self.param_vectors = param_vectors
        
        if epsilon_range is None:
            self.epsilon_range = jnp.logspace(-7, -1, 20)  # 1e-7 to 1e-1
        else:
            self.epsilon_range = epsilon_range

        self.jax_timing_runs = max(1, int(jax_timing_runs))
        
        self.results = {}
        logger.info(f"GradientBenchmark initialized: {len(self.epsilon_range)} epsilon values, {len(self.param_vectors)} param vectors")
    
    def _compute_fd_gradient(
        self,
        theta: jax.Array,
        epsilon: float
    ) -> jax.Array:
        grad_fd = jnp.zeros_like(theta)
        for i in range(len(theta)):
            theta_plus = theta.at[i].add(epsilon)
            theta_minus = theta.at[i].add(-epsilon)
            grad_fd = grad_fd.at[i].set(
                (self.objective(theta_plus) - self.objective(theta_minus)) / (2 * epsilon)
            )
        return grad_fd
    
    def sweep_epsilon(self, theta: jax.Array) -> dict:
        logger.info(f"Sweeping epsilon for theta={theta}...")

        # JAX timing decomposition: compile+first call vs steady-state execution.
        t0 = time.perf_counter()
        grad_jax_val = self.grad_jax(theta)
        grad_jax_val = jax.block_until_ready(grad_jax_val)
        time_jax_compile_first = time.perf_counter() - t0

        exec_samples = []
        for _ in range(self.jax_timing_runs):
            t1 = time.perf_counter()
            grad_jax_exec = self.grad_jax(theta)
            grad_jax_exec = jax.block_until_ready(grad_jax_exec)
            exec_samples.append(time.perf_counter() - t1)
            grad_jax_val = grad_jax_exec

        time_jax_exec_mean = float(np.mean(exec_samples))
        time_jax_exec_std = float(np.std(exec_samples))
        logger.debug(
            "JAX gradient timing: compile+first=%.3f ms, exec_mean=%.3f ms (runs=%d)",
            time_jax_compile_first * 1e3,
            time_jax_exec_mean * 1e3,
            self.jax_timing_runs,
        )
        
        epsilon_results = {}
        
        for eps in self.epsilon_range:
            t0 = time.time()
            grad_fd = self._compute_fd_gradient(theta, float(eps))
            time_fd = time.time() - t0
            
            abs_error = jnp.linalg.norm(grad_jax_val - grad_fd)
            rel_error = abs_error / (jnp.linalg.norm(grad_fd) + 1e-12)
            
            epsilon_results[float(eps)] = {
                'grad_fd': grad_fd,
                'abs_error': float(abs_error),
                'rel_error': float(rel_error),
                'time_fd': time_fd
            }
        
        return {
            'grad_jax': grad_jax_val,
            'time_jax': time_jax_exec_mean,
            'time_jax_compile_first': time_jax_compile_first,
            'time_jax_exec_mean': time_jax_exec_mean,
            'time_jax_exec_std': time_jax_exec_std,
            'epsilon_sweep': epsilon_results
        }
    
    def run_all(self) -> dict:
        logger.info(f"Running gradient benchmark on {len(self.param_vectors)} parameter vectors...")
        results = {}
        
        for idx, theta in enumerate(self.param_vectors):
            logger.info(f"Testing parameter vector {idx+1}/{len(self.param_vectors)}: {theta}")
            results[idx] = self.sweep_epsilon(theta)
        
        self.results = results
        return results
    
    def plot_convergence(self, param_idx: int = 0):
        if param_idx not in self.results:
            logger.warning(f"No results for parameter index {param_idx}")
            return
        
        sweep = self.results[param_idx]['epsilon_sweep']
        epsilons = sorted(sweep.keys())
        rel_errors = [sweep[eps]['rel_error'] for eps in epsilons]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.loglog(epsilons, rel_errors, 'o-', linewidth=2, markersize=8, color='steelblue', label='Relative Error')
        
        # Add reference lines for convergence rates
        # For central differences: error ~ O(epsilon^2)
        eps_ref = jnp.array(epsilons)
        line_o2 = rel_errors[0] * (eps_ref / epsilons[0])**2
        line_o4 = rel_errors[0] * (eps_ref / epsilons[0])**4
        
        ax.loglog(epsilons, line_o2, '--', linewidth=1.5, alpha=0.6, label=r'O($\epsilon^2$) reference')
        ax.loglog(epsilons, line_o4, ':', linewidth=1.5, alpha=0.6, label=r'O($\epsilon^4$) reference')
        
        ax.set_xlabel('Step Size eps', fontsize=12)
        ax.set_ylabel(r'Relative Error $\|\nabla_{\mathrm{JAX}} - \nabla_{\mathrm{FD}}\| / \|\nabla_{\mathrm{FD}}\|$', fontsize=12)
        ax.set_title(f'FD Convergence for theta={self.param_vectors[param_idx]}', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'result/gradient_convergence_param_{param_idx}.png', dpi=150)
        logger.info(f"Saved to result/gradient_convergence_param_{param_idx}.png")
        plt.close()
    
    def plot_all_convergence(self):
        """Plot convergence for all parameter vectors in a grid."""
        n_params = len(self.results)
        if n_params == 0:
            logger.warning("No results to plot")
            return
        
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx in range(n_params):
            sweep = self.results[idx]['epsilon_sweep']
            epsilons = sorted(sweep.keys())
            rel_errors = [sweep[eps]['rel_error'] for eps in epsilons]
            
            axes[idx].loglog(epsilons, rel_errors, 'o-', linewidth=2, markersize=6)
            axes[idx].set_xlabel('Step Size eps')
            axes[idx].set_ylabel('Relative Error')
            axes[idx].set_title(f'theta = {self.param_vectors[idx]}')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('result/gradient_convergence_all.png', dpi=150)
        logger.info("Saved to result/gradient_convergence_all.png")
        plt.close()
    
    def plot_timing_comparison(self):
        """Compare JAX vs FD computation time."""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: Time vs epsilon
        for param_idx in self.results.keys():
            sweep = self.results[param_idx]['epsilon_sweep']
            epsilons = sorted(sweep.keys())
            times_fd = [sweep[eps]['time_fd'] for eps in epsilons]
            
            axes[0].loglog(epsilons, times_fd, 'o-', linewidth=2, markersize=6, label=f'Param {param_idx}')
        
        axes[0].set_xlabel('Step Size eps')
        axes[0].set_ylabel('Wall Time (seconds)')
        axes[0].set_title('FD Computation Time vs eps')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Panel 2: JAX timing decomposition vs FD timing bar chart
        time_jax_exec_vals = [self.results[idx]['time_jax_exec_mean'] for idx in self.results.keys()]
        time_jax_compile_vals = [self.results[idx]['time_jax_compile_first'] for idx in self.results.keys()]
        time_fd_vals = []
        for param_idx in self.results.keys():
            sweep = self.results[param_idx]['epsilon_sweep']
            avg_time_fd = np.mean([sweep[eps]['time_fd'] for eps in sweep.keys()])
            time_fd_vals.append(avg_time_fd)

        x = np.arange(len(time_jax_exec_vals))
        width = 0.25

        axes[1].bar(x - width, time_jax_exec_vals, width, label='JAX AD (exec mean)', color='steelblue')
        axes[1].bar(x, time_fd_vals, width, label='FD (avg)', color='coral')
        axes[1].bar(x + width, time_jax_compile_vals, width, label='JAX compile+first', color='slategray')
        
        axes[1].set_ylabel('Wall Time (seconds)')
        axes[1].set_title('JAX AD Timing Decomposition vs FD')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'Param {i}' for i in range(len(time_jax_exec_vals))])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('result/gradient_timing_comparison.png', dpi=150)
        logger.info("Saved to result/gradient_timing_comparison.png")
        plt.close()
    
    def save_results(self, filename: str = "result/gradient_benchmark_results.npz"):
        if not self.results:
            logger.warning("No results to save")
            return
        
        data = {}
        for param_idx, param_results in self.results.items():
            data[f'param_{param_idx}_time_jax'] = param_results['time_jax']
            data[f'param_{param_idx}_time_jax_compile_first'] = param_results['time_jax_compile_first']
            data[f'param_{param_idx}_time_jax_exec_mean'] = param_results['time_jax_exec_mean']
            data[f'param_{param_idx}_time_jax_exec_std'] = param_results['time_jax_exec_std']
            sweep = param_results['epsilon_sweep']
            
            data[f'param_{param_idx}_epsilons'] = np.array(sorted(sweep.keys()))
            data[f'param_{param_idx}_rel_errors'] = np.array([sweep[eps]['rel_error'] for eps in sorted(sweep.keys())])
            data[f'param_{param_idx}_abs_errors'] = np.array([sweep[eps]['abs_error'] for eps in sorted(sweep.keys())])
            data[f'param_{param_idx}_times_fd'] = np.array([sweep[eps]['time_fd'] for eps in sorted(sweep.keys())])
        
        np.savez(filename, **data)
        logger.info(f"Saved to {filename}")

# ==========================
# 3. Upwind/central scheme comparison
# ==========================
class SchemeComparisonStudy:
    def __init__(
        self,
        viscosities: jax.Array = None,
        grid: Grid = None,
        config: SimConfig = None
    ):
        if viscosities is None:
            self.viscosities = jnp.array([0.005, 0.01, 0.05])
        else:
            self.viscosities = viscosities
        
        self.schemes = ["central", "upwind"]
        self.grid = grid or make_grid(128, 1.0)
        self.config = config or CONFIG
        self.results = {}
        
        logger.info(f"SchemeComparisonStudy initialized: {len(self.schemes)} schemes, {len(self.viscosities)} viscosity values")
    
    def run_comparison(
        self,
        theta: jax.Array,
        make_solver_fn: Callable = None,   # kept for API compatibility, unused
    ) -> dict:
        results = {}

        for scheme in self.schemes:
            logger.info(f"Testing {scheme} scheme...")
            scheme_results = {}

            for nu in self.viscosities:
                logger.info(f"nu={nu:.4f}...")

                test_config = SimConfig(
                    N=self.config.N,
                    L=self.config.L,
                    nu=float(nu),
                    mu0=self.config.mu0,
                    sigma0=self.config.sigma0,
                    C_force=self.config.C_force,
                    cfl=self.config.cfl,
                    v_max=self.config.v_max,
                    t=self.config.t,
                    theta0=theta,
                    advection_scheme=scheme,
                    obj_tol=self.config.obj_tol
                )

                grid     = make_grid(test_config.N, test_config.L)
                phys     = Phys(nu=test_config.nu, mu0=test_config.mu0,
                                sigma0=test_config.sigma0, C=test_config.C_force)
                time_obj = Time(dt=test_config.dt, steps=test_config.steps)

                solver = SingleCoreNSSolver(grid, phys, time_obj, advection_scheme=scheme)
                
                # Run simulation and collect convergence history
                u0 = jnp.zeros((2, test_config.N, test_config.N))
                objective_val, u_final, convergence_hist = self._rollout_with_history(
                    solver.step, u0, theta, grid, phys, time_obj
                )
                
                # Compute metrics
                speed = jnp.sqrt(u_final[0]**2 + u_final[1]**2)
                max_speed = float(jnp.max(speed))
                ke = float(0.5 * jnp.sum(u_final[0]**2 + u_final[1]**2) * (grid.dx**2))
                
                scheme_results[float(nu)] = {
                    'objective': float(objective_val),
                    'kinetic_energy': ke,
                    'max_velocity': max_speed,
                    'convergence_history': convergence_hist,
                    'u_final': u_final,
                    'dt': test_config.dt
                }
                
                logger.debug(f"obj={objective_val:.6e}, KE={ke:.6e}, max_u={max_speed:.4f}")
            
            results[scheme] = scheme_results
        
        self.results = results
        return results
    
    def _rollout_with_history(
        self,
        step: Callable,
        u0: jax.Array,
        theta: jax.Array,
        grid: Grid,
        phys: Phys,
        time_obj: Time
    ) -> Tuple[float, jax.Array, list]:
        u = u0
        J = 0.0
        convergence_hist = []
        forcing = GaussianForcingField(grid, phys)

        for i in range(time_obj.steps):
            t  = i * time_obj.dt
            f  = forcing(theta, t)
            J += time_obj.dt * jnp.sum(u * f) * (grid.dx**2)
            
            convergence_hist.append({
                'step': i,
                'objective': float(J),
                'time': float(t),
                'max_velocity': float(jnp.max(jnp.sqrt(u[0]**2 + u[1]**2)))
            })
            
            u = step(u, theta, t)
        
        return J, u, convergence_hist
    
    def plot_convergence_comparison(self):
        if not self.results:
            logger.warning("No results to plot")
            return
        
        n_nu = len(self.viscosities)
        fig, axes = plt.subplots(1, n_nu, figsize=(5*n_nu, 4))
        
        if n_nu == 1:
            axes = [axes]
        
        for ax_idx, nu_val in enumerate(sorted(self.viscosities)):
            ax = axes[ax_idx]
            nu = float(nu_val)
            
            for scheme in self.schemes:
                if nu not in self.results[scheme]:
                    continue
                
                hist = self.results[scheme][float(nu)]['convergence_history']
                times = [h['time'] for h in hist]
                objectives = [h['objective'] for h in hist]
                
                ax.plot(times, objectives, 'o-', linewidth=2, markersize=4, label=scheme)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Cumulative Objective J')
            ax.set_title(f'Convergence: nu={nu:.4f}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('result/scheme_comparison_convergence.png', dpi=150)
        logger.info("Saved to result/scheme_comparison_convergence.png")
        plt.close()
    
    def plot_velocity_max_evolution(self):
        if not self.results:
            logger.warning("No results to plot")
            return
        
        n_nu = len(self.viscosities)
        fig, axes = plt.subplots(1, n_nu, figsize=(5*n_nu, 4))
        
        if n_nu == 1:
            axes = [axes]
        
        for ax_idx, nu_val in enumerate(sorted(self.viscosities)):
            ax = axes[ax_idx]
            nu = float(nu_val)
            
            for scheme in self.schemes:
                if nu not in self.results[scheme]:
                    continue
                
                hist = self.results[scheme][float(nu)]['convergence_history']
                times = [h['time'] for h in hist]
                max_vels = [h['max_velocity'] for h in hist]
                
                ax.plot(times, max_vels, 'o-', linewidth=2, markersize=4, label=scheme)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Max Velocity |u|')
            ax.set_title(f'Velocity Evolution: nu={nu:.4f}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('result/scheme_comparison_velocity_evolution.png', dpi=150)
        logger.info("Saved to result/scheme_comparison_velocity_evolution.png")
        plt.close()
    
    def plot_final_velocity_fields(self):
        if not self.results:
            logger.warning("No results to plot")
            return

        def _lookup_result(scheme_results: dict, nu_value: float):
            """Lookup viscosity key robustly against float32/float64 representation drift."""
            if nu_value in scheme_results:
                return scheme_results[nu_value]
            for key in scheme_results.keys():
                if np.isclose(float(key), float(nu_value), rtol=1e-8, atol=1e-8):
                    return scheme_results[key]
            return None
        
        n_nu = len(self.viscosities)
        fig, axes = plt.subplots(n_nu, 2, figsize=(10, 4*n_nu))
        
        if n_nu == 1:
            axes = axes.reshape(1, -1)

        speed_values = []
        for nu_val in sorted(self.viscosities):
            nu = float(nu_val)
            for scheme in self.schemes:
                scheme_results = self.results.get(scheme, {})
                result_entry = _lookup_result(scheme_results, nu)
                if result_entry is not None:
                    u_final = result_entry['u_final']
                    speed_values.append(np.asarray(jnp.sqrt(u_final[0]**2 + u_final[1]**2)).ravel())

        if not speed_values:
            logger.warning("No velocity-field data available for plotting")
            plt.close(fig)
            return

        vmin, vmax = _robust_color_limits(np.concatenate(speed_values))
        
        for row_idx, nu_val in enumerate(sorted(self.viscosities)):
            nu = float(nu_val)
            for col_idx, scheme in enumerate(self.schemes):
                scheme_results = self.results.get(scheme, {})
                result_entry = _lookup_result(scheme_results, nu)
                if result_entry is None:
                    continue

                u_final = result_entry['u_final']
                speed = jnp.sqrt(u_final[0]**2 + u_final[1]**2)
                
                im = axes[row_idx, col_idx].imshow(
                    speed.T, origin='lower',
                    extent=[0, self.grid.L, 0, self.grid.L],
                    cmap='viridis',
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[row_idx, col_idx].set_title(f'{scheme.capitalize()} (nu={nu:.4f})')
                axes[row_idx, col_idx].set_xlabel('x')
                axes[row_idx, col_idx].set_ylabel('y')
                plt.colorbar(im, ax=axes[row_idx, col_idx], label='Speed')
        
        plt.tight_layout()
        plt.savefig('result/scheme_comparison_velocity_fields.png', dpi=150)
        logger.info("Saved to result/scheme_comparison_velocity_fields.png")
        plt.close()
    
    def plot_metrics_summary(self):
        """Summary bar chart comparing final metrics across schemes."""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        schemes_list = sorted(self.schemes)
        nu_list = [float(nu) for nu in sorted(self.viscosities)]
        
        # Prepare data
        objectives_by_scheme = {scheme: [] for scheme in schemes_list}
        ke_by_scheme = {scheme: [] for scheme in schemes_list}
        max_vel_by_scheme = {scheme: [] for scheme in schemes_list}
        
        for scheme in schemes_list:
            for nu in nu_list:
                if nu in self.results[scheme]:
                    objectives_by_scheme[scheme].append(self.results[scheme][float(nu)]['objective'])
                    ke_by_scheme[scheme].append(self.results[scheme][float(nu)]['kinetic_energy'])
                    max_vel_by_scheme[scheme].append(self.results[scheme][float(nu)]['max_velocity'])
        
        x = np.arange(len(nu_list))
        width = 0.35
        
        # Objective plot
        for idx, scheme in enumerate(schemes_list):
            axes[0].bar(x + idx*width - width/2, objectives_by_scheme[scheme], width, label=scheme)
        axes[0].set_xlabel('Viscosity')
        axes[0].set_ylabel('Final Objective J')
        axes[0].set_title('Work Done Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f'{nu:.3f}' for nu in nu_list])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Kinetic energy plot
        for idx, scheme in enumerate(schemes_list):
            axes[1].bar(x + idx*width - width/2, ke_by_scheme[scheme], width, label=scheme)
        axes[1].set_xlabel('Viscosity')
        axes[1].set_ylabel('Kinetic Energy')
        axes[1].set_title('Final Kinetic Energy')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'{nu:.3f}' for nu in nu_list])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Max velocity plot
        for idx, scheme in enumerate(schemes_list):
            axes[2].bar(x + idx*width - width/2, max_vel_by_scheme[scheme], width, label=scheme)
        axes[2].set_xlabel('Viscosity')
        axes[2].set_ylabel('Max Velocity')
        axes[2].set_title('Peak Velocity Reached')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f'{nu:.3f}' for nu in nu_list])
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('result/scheme_comparison_metrics_summary.png', dpi=150)
        logger.info("Saved to result/scheme_comparison_metrics_summary.png")
        plt.close()
    
    def save_results(self, filename: str = "result/scheme_comparison_results.npz"):
        """Save comparison results to NPZ file.
        
        Args:
            filename: Output filename
        """
        if not self.results:
            logger.warning("No results to save")
            return
        
        data = {}
        for scheme in self.results.keys():
            for nu in self.results[scheme].keys():
                data[f'{scheme}_nu_{nu:.4f}_objective'] = self.results[scheme][nu]['objective']
                data[f'{scheme}_nu_{nu:.4f}_kinetic_energy'] = self.results[scheme][nu]['kinetic_energy']
                data[f'{scheme}_nu_{nu:.4f}_max_velocity'] = self.results[scheme][nu]['max_velocity']
                data[f'{scheme}_nu_{nu:.4f}_dt'] = self.results[scheme][nu]['dt']
        
        np.savez(filename, **data)
        logger.info(f"Saved results to {filename}")


# ==========================
# 4. Parameter Optimization Study
# ==========================
class ParameterOptimizationStudy:
    """Compare multiple Optax optimizers on the same objective."""

    def __init__(
        self,
        objective: Callable[[jax.Array], float],
        grad_jax: Callable[[jax.Array], jax.Array],
        theta0: jax.Array,
        optim_configs: list = None,
        optim_steps: int = 100,
    ):
        self.objective = objective
        self.grad_jax = grad_jax
        self.theta0 = jnp.array(theta0)
        self.optim_steps = int(optim_steps)
        self.optim_configs = optim_configs or [
            {"name": "Adam (lr=0.01)", "optimizer": optax.adam(learning_rate=0.01)},
            {"name": "Adam (lr=0.001)", "optimizer": optax.adam(learning_rate=0.001)},
            {"name": "SGD (lr=0.1)", "optimizer": optax.sgd(learning_rate=0.1)},
            {"name": "RMSprop (lr=0.01)", "optimizer": optax.rmsprop(learning_rate=0.01)},
        ]
        self.results = {}
        logger.info(
            f"ParameterOptimizationStudy initialized: {len(self.optim_configs)} optimizers, {self.optim_steps} steps"
        )

    def _run_single_optimizer(self, name: str, optimizer) -> dict:
        theta_current = jnp.array(self.theta0)
        opt_state = optimizer.init(theta_current)
        obj_initial = self.objective(theta_current)

        @jax.jit
        def update_step(carry, step_idx):
            theta, state = carry
            grad = self.grad_jax(theta)
            updates, state = optimizer.update(grad, state)
            theta = optax.apply_updates(theta, updates)
            obj_val = self.objective(theta)
            step_history = {
                'step': step_idx + 1,
                'objective': obj_val,
                'parameter': theta.copy(),
            }
            return (theta, state), step_history

        (_, _), history_outputs = jax.lax.scan(
            update_step,
            (theta_current, opt_state),
            jnp.arange(self.optim_steps),
        )

        history = {
            'steps': jnp.concatenate([jnp.array([0]), history_outputs['step']]),
            'objectives': jnp.concatenate([jnp.array([obj_initial]), history_outputs['objective']]),
            'parameters': jnp.concatenate([jnp.expand_dims(self.theta0, axis=0), history_outputs['parameter']]),
        }

        obj_final = float(history['objectives'][-1])
        improvement = float((obj_initial - obj_final) / (obj_initial + 1e-20))
        logger.info(f"{name}: initial={float(obj_initial):.6e}, final={obj_final:.6e}, improvement={improvement*100:.3f}%")

        return {
            'obj_initial': float(obj_initial),
            'obj_final': obj_final,
            'improvement': improvement,
            'history': {
                'steps': np.array(history['steps']),
                'objectives': np.array(history['objectives']),
                'parameters': np.array(history['parameters']),
            },
        }

    def run_all(self) -> dict:
        logger.info(
            f"Testing {len(self.optim_configs)} optimizer configurations over {self.optim_steps} steps"
        )
        results = {}
        for opt_config in self.optim_configs:
            name = opt_config['name']
            optimizer = opt_config['optimizer']
            results[name] = self._run_single_optimizer(name, optimizer)
        self.results = results
        return results

    def save_results(self, filename: str = "result/optimization_study_results.npz"):
        if not self.results:
            logger.warning("No optimization results to save")
            return

        data = {}
        for opt_name, results in self.results.items():
            data[f'{opt_name}_obj_initial'] = results['obj_initial']
            data[f'{opt_name}_obj_final'] = results['obj_final']
            data[f'{opt_name}_improvement'] = results['improvement']
            data[f'{opt_name}_steps'] = np.array(results['history']['steps'])
            data[f'{opt_name}_objectives'] = np.array(results['history']['objectives'])
            data[f'{opt_name}_parameters'] = np.array(results['history']['parameters'])

        np.savez(filename, **data)
        logger.info(f"Saved to {filename}")

    def plot_convergence(self, filename: str = "result/optimization_study_convergence.png"):
        if not self.results:
            logger.warning("No optimization results to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for opt_name, results in self.results.items():
            hist = results['history']
            ax1.semilogy(hist['steps'], hist['objectives'], 'o-', linewidth=2, markersize=6, label=opt_name)

        ax1.set_xlabel('Optimization Step')
        ax1.set_ylabel('Objective J (log scale)')
        ax1.set_title('Convergence Comparison: Different Optimizers')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        opt_names = list(self.results.keys())
        final_objs = [self.results[name]['obj_final'] for name in opt_names]
        improvements = [self.results[name]['improvement'] * 100 for name in opt_names]
        x = np.arange(len(opt_names))
        ax2_twin = ax2.twinx()

        ax2.bar(x - 0.2, final_objs, 0.4, label='Final Objective', color='steelblue', alpha=0.8)
        ax2_twin.plot(x, improvements, 'ro-', linewidth=2, markersize=8, label='Improvement %')

        ax2.set_ylabel('Final Objective J', color='steelblue')
        ax2_twin.set_ylabel('Improvement (%)', color='red')
        ax2.set_xlabel('Optimizer')
        ax2.set_title('Optimization Results Summary')
        ax2.set_xticks(x)
        ax2.set_xticklabels([name.split('(')[0].strip() for name in opt_names], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        logger.info(f"Saved to {filename}")
        plt.close()


def main():
    """Main function to run all research studies."""
    import argparse
    parser = argparse.ArgumentParser(description="NS Solver Research Studies")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON config file. Missing fields fall back to defaults."
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("RESEARCH STUDIES STANDALONE")
    logger.info("=" * 80)

    config = load_config(args.config)
    grid   = make_grid(config.N, L=config.L)
    phys   = Phys(nu=config.nu, mu0=config.mu0, sigma0=config.sigma0, C=config.C_force)
    time   = Time(dt=config.dt, steps=config.steps)

    theta_opt = config.theta0
    logger.info(f"Using parameters: theta = {theta_opt}")

    # Build single-core solver for studies that need objective / gradient
    solver      = SingleCoreNSSolver(grid, phys, time, advection_scheme=config.advection_scheme)
    obj_single  = solver.jit_objective
    grad_single = solver.jit_gradient
    
    # ==========================
    # 1. Viscosity Regime Study
    # ==========================
    logger.info("\n" + "=" * 80)
    logger.info("STUDY 1: VISCOSITY REGIME ANALYSIS")
    logger.info("=" * 80)
    
    visc_study = ViscosityStudy(
        viscosities=jnp.array([0.001, 0.005, 0.01, 0.05, 0.1]),
        grid=grid,
        theta=theta_opt,
        config=config
    )
    
    logger.info("Running sweep across 5 viscosity values...")
    visc_study.run_sweep()
    
    logger.info("Generating visualizations...")
    visc_study.plot_velocity_sweep(figsize=(18, 4))
    visc_study.plot_vorticity_sweep(figsize=(18, 4))
    visc_study.plot_metrics_comparison()
    visc_study.save_results("result/viscosity_study_results.npz")
    
    logger.info("Viscosity study completed!")
    
    # ==========================
    # 2. Gradient Research
    # ==========================
    logger.info("\n" + "=" * 80)
    logger.info("STUDY 2: GRADIENT BENCHMARK (JAX AD vs FD)")
    logger.info("=" * 80)
    
    grad_bench = GradientBenchmark(
        objective=obj_single,
        grad_jax=grad_single,
        param_vectors=[
            theta_opt,
            jnp.array([0.5, 0.5]),
            jnp.array([2.0, 1.0])
        ],
        epsilon_range=jnp.logspace(-7, -1, 20),
        config=config
    )
    
    logger.info("Running benchmark across 3 parameter vectors and 20 epsilon values...")
    grad_bench.run_all()
    
    logger.info("Generating convergence and timing visualizations...")
    grad_bench.plot_all_convergence()
    grad_bench.plot_timing_comparison()
    for idx in range(len(grad_bench.param_vectors)):
        grad_bench.plot_convergence(param_idx=idx)
    grad_bench.save_results("result/gradient_benchmark_results.npz")
    
    logger.info("Gradient research completed!")
    
    # ==========================
    # 3. Scheme Comparison Study
    # ==========================
    logger.info("\n" + "=" * 80)
    logger.info("STUDY 3: SCHEME COMPARISON (CENTRAL vs UPWIND)")
    logger.info("=" * 80)
    
    scheme_study = SchemeComparisonStudy(
        viscosities=jnp.array([0.005, 0.01, 0.05]),
        grid=grid,
        config=config
    )
    
    logger.info("Running scheme comparison across 3 viscosity regimes...")
    scheme_study.run_comparison(theta_opt)
    
    logger.info("Generating comparison visualizations...")
    scheme_study.plot_convergence_comparison()
    scheme_study.plot_velocity_max_evolution()
    scheme_study.plot_final_velocity_fields()
    scheme_study.plot_metrics_summary()
    scheme_study.save_results("result/scheme_comparison_results.npz")
    
    logger.info("Scheme comparison completed!")
    
    # ==========================
    # 4. PARAMETER OPTIMIZATION STUDY
    # ==========================
    logger.info("\n" + "=" * 80)
    logger.info("STUDY 4: PARAMETER OPTIMIZATION WITH OPTAX")
    logger.info("=" * 80)

    optim_study = ParameterOptimizationStudy(
        objective=obj_single,
        grad_jax=grad_single,
        theta0=theta_opt,
        optim_steps=100,
    )
    optim_study.run_all()
    optim_study.save_results("result/optimization_study_results.npz")
    optim_study.plot_convergence("result/optimization_study_convergence.png")
    logger.info("Parameter optimization study completed!")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
