"""Comparison utilities for multiple simulation results.

Provides tools for comparing Fokker-Planck PDEs vs SDE/deterministic paths,
computing distribution metrics, and analyzing ensemble statistics.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    pass

Array = np.ndarray


def paths_to_histogram(
    paths: Array,
    grid: Any,
    time_idx: int = -1,
    alive_mask: Optional[Array] = None,
) -> Array:
    """Convert path positions to a histogram on the grid.

    Parameters
    ----------
    paths : ndarray
        Path array shape (n_paths, n_times, dim)
    grid : Grid
        Target grid for histogram
    time_idx : int
        Time index to extract (default -1 for final time)
    alive_mask : ndarray, optional
        Boolean mask (n_paths, n_times) of active paths

    Returns
    -------
    ndarray
        Histogram matching grid.shape, normalized as a probability distribution
    """
    positions = paths[:, time_idx, :]

    if alive_mask is not None:
        active = alive_mask[:, time_idx]
        positions = positions[active]

    if positions.shape[0] == 0:
        return np.zeros(grid.shape)

    # Build bin edges for each dimension
    edges = []
    for ax in grid.axis_names:
        if ax == "x":
            grid_ax = grid.x_grid
        elif ax == "y":
            grid_ax = grid.y_grid
        elif ax == "z":
            grid_ax = grid.z_grid
        else:
            raise ValueError(f"Unknown axis '{ax}'")

        # Midpoint edges
        dx = grid_ax[1] - grid_ax[0] if len(grid_ax) > 1 else 1.0
        edge = np.concatenate(
            [[grid_ax[0] - dx / 2], (grid_ax[:-1] + grid_ax[1:]) / 2, [grid_ax[-1] + dx / 2]]
        )
        edges.append(edge)

    # Compute histogram
    hist, _ = np.histogramdd(positions, bins=edges)

    # Normalize to probability distribution
    hist = hist / (np.sum(hist) * grid.volume_element)

    result: np.ndarray = hist
    return result


def compare_distributions(
    f_fp: Array,
    f_paths: Array,
    grid: Any,
) -> Dict[str, float]:
    """Compare Fokker-Planck and path-based distributions.

    Parameters
    ----------
    f_fp : ndarray
        FP distribution on grid
    f_paths : ndarray
        Path-based histogram on grid
    grid : Grid
        Spatial grid

    Returns
    -------
    dict
        Comparison metrics: 'l1_error', 'l2_error', 'kl_divergence', 'wasserstein_1d'
    """
    dV = grid.volume_element

    # L1 error
    l1_error = np.sum(np.abs(f_fp - f_paths)) * dV

    # L2 error
    l2_error = np.sqrt(np.sum((f_fp - f_paths) ** 2) * dV)

    # KL divergence D_KL(f_fp || f_paths)
    f_fp_safe = np.where(f_fp > 1e-15, f_fp, 1e-15)
    f_paths_safe = np.where(f_paths > 1e-15, f_paths, 1e-15)
    kl_div = np.sum(f_fp_safe * np.log(f_fp_safe / f_paths_safe)) * dV

    # 1D Wasserstein distance (only for 1D)
    if len(grid.axis_names) == 1:
        from scipy.stats import wasserstein_distance

        x = grid.x_grid
        wasserstein = wasserstein_distance(x, x, f_fp.flatten(), f_paths.flatten())
    else:
        wasserstein = np.nan

    return {
        "l1_error": l1_error,
        "l2_error": l2_error,
        "kl_divergence": kl_div,
        "wasserstein_1d": wasserstein,
    }


def compare_moments(
    f_fp: Array,
    grid: Any,
    mean_paths: Array,
    var_paths: Array,
) -> Dict[str, Any]:
    """Compare moments from FP distribution vs path statistics.

    Parameters
    ----------
    f_fp : ndarray
        FP distribution
    grid : Grid
        Spatial grid
    mean_paths : ndarray
        Mean from paths, shape (dim,)
    var_paths : ndarray
        Variance from paths, shape (dim,)

    Returns
    -------
    dict
        Contains 'mean_fp', 'var_fp', 'mean_error', 'var_error'
    """
    dV = grid.volume_element

    # Compute FP moments
    dim = len(grid.axis_names)
    mean_fp = np.zeros(dim)
    var_fp = np.zeros(dim)

    for i, ax in enumerate(grid.axis_names):
        if ax == "x":
            coords = grid.X
        elif ax == "y":
            coords = grid.Y
        elif ax == "z":
            coords = grid.Z
        else:
            raise ValueError(f"Unknown axis '{ax}'")

        mean_fp[i] = np.sum(coords * f_fp) * dV
        var_fp[i] = np.sum((coords - mean_fp[i]) ** 2 * f_fp) * dV

    return {
        "mean_fp": mean_fp,
        "var_fp": var_fp,
        "mean_error": np.abs(mean_fp - mean_paths),
        "var_error": np.abs(var_fp - var_paths),
        "mean_rel_error": np.abs(mean_fp - mean_paths) / (np.abs(mean_fp) + 1e-12),
        "var_rel_error": np.abs(var_fp - var_paths) / (np.abs(var_fp) + 1e-12),
    }


class ResultComparison:
    """Compare multiple SimulationResult objects.

    Enables side-by-side analysis of FP, SDE, and deterministic solutions.
    """

    def __init__(self, results: List[Any], labels: Optional[List[str]] = None):
        """Initialize comparison.

        Parameters
        ----------
        results : list of SimulationResult
            Results to compare
        labels : list of str, optional
            Custom labels for each result. If None, uses solver_type.
        """
        if len(results) < 2:
            raise ValueError("Need at least 2 results to compare")

        self.results = results
        self.labels = labels or [r.solver_type for r in results]

        if len(self.labels) != len(self.results):
            raise ValueError("Number of labels must match number of results")

    def differences(self) -> Dict[str, np.ndarray]:
        """Compute pairwise differences between results.

        Returns
        -------
        dict
            Keys like '0-1', '0-2', etc. with difference metrics
        """
        raise NotImplementedError("differences() in development")

    def ensemble_agreement(self) -> Dict[str, float]:
        """Check agreement between solvers (FP vs SDE variance, etc.).

        Returns
        -------
        dict
            Metrics like 'wasserstein_distance', 'mean_absolute_error', etc.
        """
        raise NotImplementedError("ensemble_agreement() in development")

    def plot_comparison_1d(self, axis: int = 0, time_idx: Optional[int] = None):
        """Compare 1D marginals across solvers.

        Parameters
        ----------
        axis : int
            Which spatial axis to marginalize over (0, 1, 2)
        time_idx : int, optional
            Which time index to show. If None, shows final time.
        """
        raise NotImplementedError("plot_comparison_1d() in development")

    def plot_comparison_2d(self, axes: tuple = (0, 1), time_idx: Optional[int] = None):
        """Compare 2D slices across solvers.

        Parameters
        ----------
        axes : tuple
            Which two spatial axes to show
        time_idx : int, optional
            Which time index. If None, shows final time.
        """
        raise NotImplementedError("plot_comparison_2d() in development")

    def summary_table(self) -> Dict[str, Dict[str, Any]]:
        """Generate summary statistics table for all results.

        Returns
        -------
        dict
            Keys are solver labels, values are dicts of statistics
        """
        summary = {}
        for label, result in zip(self.labels, self.results):
            summary[label] = {
                "solver_type": result.solver_type,
                "t_final": result.t_final,
                "duration": result.duration,
                "n_steps": result.n_steps,
                "grid_shape": (result.grid.shape if hasattr(result.grid, "shape") else None),
            }
        return summary
