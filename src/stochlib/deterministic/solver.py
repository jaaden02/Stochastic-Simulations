"""Deterministic PDE solver for advection equations without diffusion.

This module provides deterministic numerical methods for solving advection PDEs
of the form: ∂u/∂t + ∂(v·u)/∂x = 0

Includes automatic scheme selection based on CFL condition and problem parameters.
"""

from __future__ import annotations

from typing import Optional, Callable, Literal, TYPE_CHECKING
import numpy as np
import warnings
from ..logging_utils import get_logger

if TYPE_CHECKING:
    from ..setup import Grid, InitialCondition

logger = get_logger("deterministic.solver")
SchemeType = Literal["auto", "upwind", "lax_wendroff", "beam_warming"]


class DeterministicPDESolver:
    """Solver for deterministic advection PDEs without diffusion.

    Automatically selects appropriate numerical scheme based on:
    - CFL stability condition
    - Grid resolution
    - Drift velocity characteristics

    Attributes
    ----------
    grid : Grid
        Spatial grid
    ic : InitialCondition
        Initial condition
    drift : callable
        Drift velocity function v(x) or v(x, t)
    scheme : str
        Numerical scheme: 'upwind', 'lax_wendroff', 'beam_warming', or 'auto'
    boundary_mode : str
        Boundary handling: 'none', 'periodic', 'absorb', 'reflect'
    bounds : tuple or None
        Spatial bounds for boundary conditions
    """

    def __init__(
        self,
        grid: "Grid",
        ic: "InitialCondition",
        drift: Optional[Callable] = None,
        scheme: SchemeType = "auto",
        boundary_mode: str = "none",
        bounds: Optional[tuple] = None,
    ):
        """Initialize deterministic PDE solver.

        Parameters
        ----------
        grid : Grid
            Spatial grid
        ic : InitialCondition
            Initial condition
        drift : callable, optional
            Drift velocity function v(x) or v(x, t). If None, no advection.
        scheme : str
            Numerical scheme selection
        boundary_mode : str
            Boundary condition mode
        bounds : tuple, optional
            (lower, upper) spatial bounds for boundaries
        """
        self.grid = grid
        self.ic = ic
        self.drift = drift
        self.scheme = scheme
        self.boundary_mode = boundary_mode
        self.bounds = bounds

        # Initialize solution
        self.u = ic.f0.copy()
        self.t_current = 0.0

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate solver inputs."""
        if self.boundary_mode not in ["none", "periodic", "absorb", "reflect"]:
            raise ValueError(
                f"Unknown boundary_mode: '{self.boundary_mode}'. "
                f"Valid modes: 'none', 'periodic', 'absorb', 'reflect'. "
                f"- 'none': No boundary conditions. "
                f"- 'periodic': Periodic boundary conditions. "
                f"- 'absorb': Absorbing boundaries (requires bounds). "
                f"- 'reflect': Reflecting boundaries (requires bounds)."
            )

        if self.boundary_mode in ["absorb", "reflect"] and self.bounds is None:
            raise ValueError(
                f"boundary_mode='{self.boundary_mode}' requires bounds parameter. "
                f"Pass bounds as tuple (lower, upper), e.g., bounds=(-10.0, 10.0)"
            )

        if self.scheme not in ["auto", "upwind", "lax_wendroff", "beam_warming"]:
            raise ValueError(
                f"Unknown scheme: '{self.scheme}'. "
                f"Valid schemes: 'auto', 'upwind', 'lax_wendroff', 'beam_warming'. "
                f"'auto' selects based on grid resolution and stability."
            )

    def choose_scheme(self, dt: float) -> str:
        """Automatically choose numerical scheme based on problem parameters.

        Parameters
        ----------
        dt : float
            Time step size

        Returns
        -------
        scheme : str
            Selected numerical scheme
        """
        if self.scheme != "auto":
            return self.scheme

        # Compute CFL number
        if self.drift is not None:
            v_max = np.max(np.abs([self.drift(x) for x in self.grid.X]))
        else:
            v_max = 0.0

        dx = self.grid.dx
        cfl = v_max * dt / dx if dx > 0 else 0.0

        # Select scheme based on CFL and accuracy requirements
        if cfl > 1.0:
            warnings.warn(
                f"CFL condition violated: CFL={cfl:.3f} > 1.0. "
                f"Consider reducing dt or increasing dx. Using upwind scheme.",
                UserWarning,
            )
            return "upwind"
        elif cfl > 0.5:
            # Use stable first-order upwind
            return "upwind"
        else:
            # Use higher-order Lax-Wendroff for better accuracy
            return "lax_wendroff"

    def _apply_upwind(self, u: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
        """Apply upwind scheme for advection.

        First-order accurate, unconditionally stable for CFL ≤ 1.
        """
        u_new = u.copy()
        dx = self.grid.dx
        n_grid = len(u)

        for j in range(n_grid):
            if v[j] > 0 and j > 0:
                # Upwind from left
                u_new[j] = u[j] - (dt / dx) * v[j] * (u[j] - u[j - 1])
            elif v[j] < 0 and j < n_grid - 1:
                # Upwind from right
                u_new[j] = u[j] - (dt / dx) * v[j] * (u[j + 1] - u[j])

        return u_new

    def _apply_lax_wendroff(
        self, u: np.ndarray, v: np.ndarray, dt: float
    ) -> np.ndarray:
        """Apply Lax-Wendroff scheme for advection.

        Second-order accurate in space and time, stable for CFL ≤ 1.
        """
        u_new = u.copy()
        dx = self.grid.dx
        n_grid = len(u)

        for j in range(1, n_grid - 1):
            # Lax-Wendroff: second-order central difference with diffusion correction
            flux_plus = 0.5 * v[j] * (u[j + 1] + u[j]) - 0.5 * (v[j] * dt / dx) * v[
                j
            ] * (u[j + 1] - u[j])
            flux_minus = 0.5 * v[j - 1] * (u[j] + u[j - 1]) - 0.5 * (
                v[j - 1] * dt / dx
            ) * v[j - 1] * (u[j] - u[j - 1])
            u_new[j] = u[j] - (dt / dx) * (flux_plus - flux_minus)

        return u_new

    def _apply_beam_warming(
        self, u: np.ndarray, v: np.ndarray, dt: float
    ) -> np.ndarray:
        """Apply Beam-Warming scheme for advection.

        Second-order upwind scheme, good for smooth solutions.
        """
        u_new = u.copy()
        dx = self.grid.dx
        n_grid = len(u)

        for j in range(2, n_grid - 1):
            if v[j] > 0:
                # Second-order upwind from left
                u_new[j] = u[j] - (dt / dx) * v[j] * (
                    1.5 * u[j] - 2 * u[j - 1] + 0.5 * u[j - 2]
                )
            elif v[j] < 0:
                # Second-order upwind from right
                u_new[j] = u[j] - (dt / dx) * v[j] * (
                    -1.5 * u[j] + 2 * u[j + 1] - 0.5 * u[j + 2]
                )

        return u_new

    def _apply_boundary_conditions(self, u: np.ndarray) -> np.ndarray:
        """Apply boundary conditions to solution."""
        if self.boundary_mode == "periodic":
            u[0] = u[-2]
            u[-1] = u[1]
        elif self.boundary_mode == "absorb" and self.bounds is not None:
            mask = (self.grid.X < self.bounds[0]) | (self.grid.X > self.bounds[1])
            u[mask] = 0.0
        elif self.boundary_mode == "reflect" and self.bounds is not None:
            # Zero-flux (Neumann) at boundaries
            u[self.grid.X < self.bounds[0]] = 0.0
            u[self.grid.X > self.bounds[1]] = 0.0

        return u

    def step(self, dt: float, t: Optional[float] = None) -> np.ndarray:
        """Advance solution by one time step.

        Parameters
        ----------
        dt : float
            Time step size
        t : float, optional
            Current time (for time-dependent drift)

        Returns
        -------
        u : ndarray
            Updated solution
        """
        if t is None:
            t = self.t_current

        # Get drift velocities
        if self.drift is not None:
            try:
                # Try time-dependent drift v(x, t)
                v = np.array([self.drift(x, t) for x in self.grid.X])
            except TypeError:
                # Fall back to time-independent v(x)
                v = np.array([self.drift(x) for x in self.grid.X])
        else:
            v = np.zeros_like(self.grid.X)

        # Choose and apply numerical scheme
        scheme = self.choose_scheme(dt)

        if scheme == "upwind":
            u_new = self._apply_upwind(self.u, v, dt)
        elif scheme == "lax_wendroff":
            u_new = self._apply_lax_wendroff(self.u, v, dt)
        elif scheme == "beam_warming":
            u_new = self._apply_beam_warming(self.u, v, dt)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        # Apply boundary conditions
        u_new = self._apply_boundary_conditions(u_new)

        # Preserve mass if no absorbing boundaries
        if self.boundary_mode in ["none", "periodic"]:
            mass_old = np.sum(self.u) * self.grid.dx
            mass_new = np.sum(u_new) * self.grid.dx
            if mass_new > 0 and mass_old > 0:
                u_new = u_new * (mass_old / mass_new)

        self.u = u_new
        self.t_current = t + dt

        return self.u.copy()

    def solve(self, t_eval: np.ndarray, verbose: bool = False) -> tuple:
        """Solve PDE over time interval.

        Parameters
        ----------
        t_eval : ndarray
            Time points at which to evaluate solution
        verbose : bool
            Print progress information

        Returns
        -------
        (t_eval, u_history) : tuple
            Time points and solution history (n_times, n_grid)
        """
        # Reset to initial condition
        self.u = self.ic.f0.copy()
        self.t_current = t_eval[0]

        u_history = [self.u.copy()]
        dt = np.mean(np.diff(t_eval))

        # Get scheme name for reporting
        scheme_name = self.choose_scheme(dt)

        if verbose:
            print(f"Solving deterministic PDE with {scheme_name} scheme")
            print(f"Grid: {len(self.grid.X)} points, dx={self.grid.dx:.4f}")
            print(f"Time: {len(t_eval)} steps, dt={dt:.4f}")

            if self.drift is not None:
                v_test = np.array([self.drift(x) for x in self.grid.X])
                v_max = np.max(np.abs(v_test))
                cfl = v_max * dt / self.grid.dx
                print(f"CFL number: {cfl:.4f}")

        for i in range(1, len(t_eval)):
            self.step(dt, t=t_eval[i - 1])
            u_history.append(self.u.copy())

        return t_eval, np.array(u_history)

    def reset(self):
        """Reset solver to initial condition."""
        self.u = self.ic.f0.copy()
        self.t_current = 0.0


def solve_deterministic_pde(
    grid: "Grid",
    ic: "InitialCondition",
    t_eval: np.ndarray,
    drift: Optional[Callable] = None,
    scheme: SchemeType = "auto",
    boundary_mode: str = "none",
    bounds: Optional[tuple] = None,
    verbose: bool = False,
) -> tuple:
    """Solve deterministic advection PDE.

    Convenience function for one-shot solving without creating solver object.

    Parameters
    ----------
    grid : Grid
        Spatial grid
    ic : InitialCondition
        Initial condition
    t_eval : ndarray
        Time points for solution
    drift : callable, optional
        Drift velocity function
    scheme : str
        Numerical scheme ('auto', 'upwind', 'lax_wendroff', 'beam_warming')
    boundary_mode : str
        Boundary condition mode
    bounds : tuple, optional
        Spatial bounds
    verbose : bool
        Print progress information

    Returns
    -------
    (t_eval, u) : tuple
        Time points and solution array (n_times, n_grid)
    """
    solver = DeterministicPDESolver(
        grid=grid,
        ic=ic,
        drift=drift,
        scheme=scheme,
        boundary_mode=boundary_mode,
        bounds=bounds,
    )

    return solver.solve(t_eval, verbose=verbose)
