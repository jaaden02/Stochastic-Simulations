"""Runtime diagnostics for SDE path simulations."""

from typing import Dict, List, Optional
import numpy as np
from ..logging_utils import get_logger

logger = get_logger("sde.diagnostic")
Array = np.ndarray


class PathDiagnostics:
    """Track simple statistics for many-path simulations."""

    def __init__(self) -> None:
        self.history: Dict[str, List] = {
            "mean": [],
            "var": [],
            "min": [],
            "max": [],
            "stderr": [],
            "n_paths": [],
        }

    def record(self, x: Array, t: float, alive_mask: Optional[Array] = None) -> None:
        """Record statistics at a time step.

        Parameters
        ----------
        x : ndarray
            Current path positions, shape (n_paths, dim)
        t : float
            Current time
        alive_mask : ndarray, optional
            Boolean mask of active paths. If None, all paths are active.
        """
        if alive_mask is not None:
            x_active = x[alive_mask]
            n_active = np.sum(alive_mask)
        else:
            x_active = x
            n_active = x.shape[0]

        if n_active == 0:
            # No active paths
            self.history["mean"].append(np.full(x.shape[1], np.nan))
            self.history["var"].append(np.full(x.shape[1], np.nan))
            self.history["min"].append(np.full(x.shape[1], np.nan))
            self.history["max"].append(np.full(x.shape[1], np.nan))
            self.history["stderr"].append(np.full(x.shape[1], np.nan))
            self.history["n_paths"].append(0)
            return

        mean = np.mean(x_active, axis=0)
        var = np.var(x_active, axis=0, ddof=1) if n_active > 1 else np.zeros(x.shape[1])
        stderr = np.sqrt(var / n_active) if n_active > 1 else np.zeros(x.shape[1])

        self.history["mean"].append(mean)
        self.history["var"].append(var)
        self.history["min"].append(np.min(x_active, axis=0))
        self.history["max"].append(np.max(x_active, axis=0))
        self.history["stderr"].append(stderr)
        self.history["n_paths"].append(n_active)

    def confidence_interval(self, confidence: float = 0.95, dim: int = 0):
        """Compute confidence intervals for the mean trajectory.

        Parameters
        ----------
        confidence : float
            Confidence level (default 0.95 for 95% CI)
        dim : int
            Dimension to compute CI for

        Returns
        -------
        dict
            Contains 'mean', 'lower', 'upper' arrays for the specified dimension
        """
        from scipy import stats

        mean_traj = np.array([m[dim] for m in self.history["mean"]])
        stderr_traj = np.array([s[dim] for s in self.history["stderr"]])
        n_paths_traj = np.array(self.history["n_paths"])

        # t-distribution critical value
        alpha = 1 - confidence
        df = n_paths_traj - 1
        t_crit = stats.t.ppf(1 - alpha / 2, df)

        margin = t_crit * stderr_traj

        return {
            "mean": mean_traj,
            "lower": mean_traj - margin,
            "upper": mean_traj + margin,
        }

    def convergence_check(self, dim: int = 0, window: int = 10) -> Dict[str, float]:
        """Check convergence of mean estimate over recent time steps.

        Parameters
        ----------
        dim : int
            Dimension to check
        window : int
            Number of recent steps to analyze

        Returns
        -------
        dict
            Contains 'relative_change', 'mean_stderr', and 'converged' flag
        """
        if len(self.history["mean"]) < window + 1:
            return {"relative_change": np.nan, "mean_stderr": np.nan, "converged": False}

        recent_means = np.array([m[dim] for m in self.history["mean"][-window:]])
        recent_stderr = np.array([s[dim] for s in self.history["stderr"][-window:]])

        # Relative change in mean
        rel_change = np.abs(recent_means[-1] - recent_means[0]) / (np.abs(recent_means[0]) + 1e-12)

        # Average standard error
        mean_stderr = np.mean(recent_stderr)

        # Heuristic: converged if relative change < 1% and stderr is small
        converged = (rel_change < 0.01) and (mean_stderr < 0.1 * np.abs(recent_means[-1]))

        return {
            "relative_change": rel_change,
            "mean_stderr": mean_stderr,
            "converged": converged,
        }
