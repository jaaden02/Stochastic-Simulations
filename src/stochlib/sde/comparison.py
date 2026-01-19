"""Comparison tools for Fokker-Planck PDE vs SDE path simulations."""
import numpy as np
from typing import Dict, Any, Optional
from ..setup import Grid

Array = np.ndarray


def paths_to_histogram(
    paths: Array,
    grid: Grid,
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
        if ax == 'x':
            grid_ax = grid.x_grid
        elif ax == 'y':
            grid_ax = grid.y_grid
        elif ax == 'z':
            grid_ax = grid.z_grid
        else:
            raise ValueError(f"Unknown axis '{ax}'")
        
        # Midpoint edges
        dx = grid_ax[1] - grid_ax[0] if len(grid_ax) > 1 else 1.0
        edge = np.concatenate([
            [grid_ax[0] - dx/2],
            (grid_ax[:-1] + grid_ax[1:]) / 2,
            [grid_ax[-1] + dx/2]
        ])
        edges.append(edge)
    
    # Compute histogram
    hist, _ = np.histogramdd(positions, bins=edges)
    
    # Normalize to probability distribution
    hist = hist / (np.sum(hist) * grid.volume_element)
    
    return hist


def compare_distributions(
    f_fp: Array,
    f_paths: Array,
    grid: Grid,
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
    l2_error = np.sqrt(np.sum((f_fp - f_paths)**2) * dV)
    
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
    grid: Grid,
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
        if ax == 'x':
            coords = grid.X
        elif ax == 'y':
            coords = grid.Y
        elif ax == 'z':
            coords = grid.Z
        else:
            raise ValueError(f"Unknown axis '{ax}'")
        
        mean_fp[i] = np.sum(coords * f_fp) * dV
        var_fp[i] = np.sum((coords - mean_fp[i])**2 * f_fp) * dV
    
    return {
        "mean_fp": mean_fp,
        "var_fp": var_fp,
        "mean_error": np.abs(mean_fp - mean_paths),
        "var_error": np.abs(var_fp - var_paths),
        "mean_rel_error": np.abs(mean_fp - mean_paths) / (np.abs(mean_fp) + 1e-12),
        "var_rel_error": np.abs(var_fp - var_paths) / (np.abs(var_fp) + 1e-12),
    }
