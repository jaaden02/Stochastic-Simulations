"""Phase portrait visualization for SDE paths."""
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_phase_portrait(
    paths: np.ndarray,
    dim_x: int = 0,
    dim_y: int = 1,
    ax: Optional[plt.Axes] = None,
    max_paths: int = 10,
) -> plt.Axes:
    """Plot 2D phase portrait for selected dimensions.

    Parameters
    ----------
    paths : ndarray
        Shape (n_paths, n_times, n_dims)
    dim_x, dim_y : int
        Dimensions to plot on x and y axes
    ax : Axes, optional
        Matplotlib axes
    max_paths : int
        Max number of paths to plot

    Returns
    -------
    ax : Axes
    """
    ax = ax or plt.gca()
    n_paths, n_times, n_dims = paths.shape
    if dim_x >= n_dims or dim_y >= n_dims:
        raise ValueError(f"dims out of range for {n_dims}-dimensional paths")
    
    n_paths_to_plot = min(n_paths, max_paths)
    for i in range(n_paths_to_plot):
        ax.plot(paths[i, :, dim_x], paths[i, :, dim_y], alpha=0.6)
        ax.scatter(paths[i, 0, dim_x], paths[i, 0, dim_y], c='green', s=20, zorder=5)
        ax.scatter(paths[i, -1, dim_x], paths[i, -1, dim_y], c='red', s=20, zorder=5)
    
    ax.set_xlabel(f"X[{dim_x}]")
    ax.set_ylabel(f"X[{dim_y}]")
    ax.set_title("Phase Portrait")
    return ax
