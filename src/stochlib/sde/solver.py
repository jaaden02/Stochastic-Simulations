"""Path simulator for stochastic differential equations."""

from typing import Callable, Optional, Dict, Any, Tuple
import numpy as np
from .kernel import euler_maruyama_step, milstein_step, normalize_callable
from .selector import StepSchemeAdvisor
from .diagnostic import PathDiagnostics
from ..logging_utils import get_logger

logger = get_logger("sde.solver")
Array = np.ndarray


class PathSimulator:
    """Simulate many sample paths for an SDE.

    The SDE is interpreted as dX_t = mu(X_t, t) dt + sigma(X_t, t) dW_t.
    Diffusion is treated as diagonal; supply ``diffusion_jacobian`` for Milstein.
    """

    def __init__(
        self,
        drift: Callable[[Array, float], Array],
        diffusion: Callable[[Array, float], Array],
        diffusion_jacobian: Optional[Callable[[Array, float], Array]] = None,
        scheme: str = "auto",
        rng: Optional[np.random.Generator] = None,
        bounds: Optional[Tuple[Array, Array]] = None,
        boundary_mode: Optional[str] = None,
    ) -> None:
        """Initialize path simulator.

        Parameters
        ----------
        drift : callable
            Drift function mu(x, t)
        diffusion : callable
            Diffusion function sigma(x, t)
        diffusion_jacobian : callable, optional
            Jacobian of diffusion for Milstein scheme
        scheme : str
            Integration scheme: "auto", "euler_maruyama", or "milstein"
        rng : Generator, optional
            Random number generator
        bounds : tuple of arrays, optional
            (lower_bounds, upper_bounds) each shape (dim,). If None, no boundary checking.
        boundary_mode : str, optional
            How to handle boundary crossings:
            - None: no boundary handling (infinite domain)
            - "absorb": path stops contributing after exit (past data valid)
            - "reject": entire path discarded if any exit occurs (all data invalid)
            - "reflect": paths bounce back elastically at boundaries
        """
        self.drift = drift
        self.diffusion = diffusion
        self.diffusion_jacobian = diffusion_jacobian
        self.scheme = StepSchemeAdvisor.choose_scheme(
            scheme=scheme,
            diffusion=diffusion,
            diffusion_jacobian=diffusion_jacobian,
        )
        self.rng = rng or np.random.default_rng()
        self.bounds = bounds
        self.boundary_mode = boundary_mode

        if boundary_mode is not None:
            if boundary_mode not in {"absorb", "reject", "reflect"}:
                raise ValueError(
                    f"boundary_mode must be 'absorb', 'reject', 'reflect', or None, got '{boundary_mode}'"
                )
            if bounds is None:
                raise ValueError("bounds required when boundary_mode is specified")

    def simulate(
        self,
        x0: Array,
        t_array: Array,
        n_paths: Optional[int] = None,
        diagnostics: Optional[PathDiagnostics] = None,
        save_paths: bool = True,
    ) -> Dict[str, Any]:
        """Run the path simulation.

        Parameters
        ----------
        x0 : array_like
            Initial state. Shape (dim,) or (n_paths, dim).
        t_array : array_like
            Monotone time grid.
        n_paths : int, optional
            Number of paths if ``x0`` is 1D. Ignored if ``x0`` already includes paths.
        diagnostics : PathDiagnostics, optional
            Runtime diagnostics collector.
        save_paths : bool
            If False, only summary stats are returned.

        Returns
        -------
        dict
            Contains:
            - mean, var: trajectories (only valid paths if boundary_mode is set)
            - times: time array
            - diagnostics: diagnostics history
            - paths: full path array if save_paths=True
            - alive_mask: boolean array (n_paths, n_times) indicating active paths
            - n_alive: number of active paths at each time (if boundaries)
            - exit_times: time when each path exited (NaN if never exited)
        """
        t_array = np.asarray(t_array, dtype=float)
        if t_array.ndim != 1 or len(t_array) < 2:
            raise ValueError(
                f"t_array must be 1D with at least two entries. "
                f"Received shape {t_array.shape} with ndim={t_array.ndim}. "
                f"Example: t_array = np.linspace(0, 1, 101)"
            )
        dt_array = np.diff(t_array)
        if np.any(dt_array <= 0):
            bad_idx = np.where(dt_array <= 0)[0]
            raise ValueError(
                f"t_array must be strictly increasing. "
                f"Found non-positive differences at indices {bad_idx.tolist()}: "
                f"t[{bad_idx[0]}]={t_array[bad_idx[0]]}, "
                f"t[{bad_idx[0]+1}]={t_array[bad_idx[0]+1]}. "
                f"Use np.sort(np.unique(t_array)) or np.linspace() to fix."
            )

        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.ndim == 1:
            if n_paths is None:
                raise ValueError(
                    f"n_paths required when x0 is 1D (shape {x0_arr.shape}). "
                    f"Pass either: n_paths=100, or x0 with shape (n_paths, dim) "
                    f"e.g., x0=np.random.normal(size=(100, {x0_arr.shape[0]}))"
                )
            x = np.broadcast_to(x0_arr, (n_paths, x0_arr.shape[0])).astype(float)
        elif x0_arr.ndim == 2:
            x = x0_arr.copy()
            n_paths = x.shape[0]
        else:
            raise ValueError(
                f"x0 must be shape (dim,) or (n_paths, dim), got shape {x0_arr.shape}. "
                f"Scalar: x0 = np.array([1.0]) (shape: (1,)). "
                f"Ensemble: x0 = np.random.normal(size=(100, 1)) (shape: (100, 1))"
            )

        dim = x.shape[1]
        stepper = self._get_stepper(dim)
        diagnostics = diagnostics or PathDiagnostics()

        # Boundary tracking
        alive = np.ones(n_paths, dtype=bool)  # all paths start alive
        ever_exited = np.zeros(
            n_paths, dtype=bool
        )  # track if path ever exited (for "reject" mode)
        exit_times = np.full(n_paths, np.nan)  # time when each path exited

        paths = None
        if save_paths:
            paths = np.zeros((n_paths, len(t_array), dim), dtype=float)
            paths[:, 0, :] = x

        # Alive mask for all time steps
        alive_mask = np.ones((n_paths, len(t_array)), dtype=bool)
        n_alive_history = [np.sum(alive)]

        # Initial statistics (all paths alive at t=0)
        mean_history = [np.mean(x, axis=0)]
        var_history = [np.var(x, axis=0)]

        for i, dt in enumerate(dt_array, start=1):
            t = t_array[i - 1]
            x = stepper(x, t, dt)

            # Check boundaries
            if self.boundary_mode is not None:
                if self.boundary_mode == "reflect":
                    # Reflect paths at boundaries
                    x = self._apply_reflection(x)
                    exited_now = np.zeros(
                        n_paths, dtype=bool
                    )  # no paths "exit" with reflection
                else:
                    exited_now = self._check_boundaries(x)
                    ever_exited |= exited_now

                    # Mark exit times
                    newly_exited = exited_now & alive
                    if np.any(newly_exited):
                        exit_times[newly_exited] = t_array[i]

                    if self.boundary_mode == "absorb":
                        # Paths stop contributing once they exit
                        alive &= ~exited_now
                    elif self.boundary_mode == "reject":
                        # Paths that ever exit are invalid
                        alive = ~ever_exited

            alive_mask[:, i] = alive
            n_alive_history.append(np.sum(alive))

            # Diagnostics on all paths (including dead ones for now)
            diagnostics.record(x, t_array[i])

            if save_paths:
                paths[:, i, :] = x

            # Statistics computed only on alive paths
            if np.any(alive):
                mean_history.append(np.mean(x[alive], axis=0))
                var_history.append(np.var(x[alive], axis=0))
            else:
                # All paths dead - use NaN
                mean_history.append(np.full(dim, np.nan))
                var_history.append(np.full(dim, np.nan))

        result = {
            "mean": np.stack(mean_history),
            "var": np.stack(var_history),
            "times": t_array,
            "diagnostics": diagnostics.history,
            "alive_mask": alive_mask,
            "n_alive": np.array(n_alive_history),
            "exit_times": exit_times,
        }
        if save_paths:
            result["paths"] = paths
        return result

    def _check_boundaries(self, x: Array) -> Array:
        """Check if any paths have crossed boundaries.

        Returns
        -------
        ndarray
            Boolean array (n_paths,) indicating which paths are outside bounds
        """
        if self.bounds is None:
            return np.zeros(x.shape[0], dtype=bool)

        lower, upper = self.bounds
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        # Check if any dimension is out of bounds
        below = np.any(x < lower, axis=1)
        above = np.any(x > upper, axis=1)
        return below | above

    def _apply_reflection(self, x: Array) -> Array:
        """Apply elastic reflection at boundaries.

        Parameters
        ----------
        x : ndarray
            Current positions, shape (n_paths, dim)

        Returns
        -------
        ndarray
            Reflected positions
        """
        if self.bounds is None:
            return x

        lower, upper = self.bounds
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        # Reflect each dimension independently
        x_reflected = x.copy()

        for d in range(x.shape[1]):
            # Lower boundary reflection
            below = x[:, d] < lower[d]
            x_reflected[below, d] = 2 * lower[d] - x[below, d]

            # Upper boundary reflection
            above = x[:, d] > upper[d]
            x_reflected[above, d] = 2 * upper[d] - x[above, d]

        return x_reflected

    def _get_stepper(self, dim: int):
        drift_fn = normalize_callable(self.drift, dim)
        diffusion_fn = normalize_callable(self.diffusion, dim)
        diffusion_jac = None
        if self.diffusion_jacobian is not None:
            diffusion_jac = normalize_callable(self.diffusion_jacobian, dim)

        if self.scheme == "milstein":

            def _step(x, t, dt):
                return milstein_step(
                    x, drift_fn, diffusion_fn, diffusion_jac, t, dt, self.rng
                )

        else:

            def _step(x, t, dt):
                return euler_maruyama_step(x, drift_fn, diffusion_fn, t, dt, self.rng)

        return _step
