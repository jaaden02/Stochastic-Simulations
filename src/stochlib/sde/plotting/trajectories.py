"""Trajectory visualization for SDE paths."""
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectories(
    paths: np.ndarray,
    times: np.ndarray,
    ax: Optional[plt.Axes] = None,
    max_paths: int = 20,
    dim: int = 0,
    dims: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
) -> plt.Axes:
    """Plot sample trajectories.

    Parameters
    ----------
    paths : ndarray
        Shape (n_paths, n_times, n_dims)
    times : ndarray
        Time points
    ax : Axes, optional
        Matplotlib axes
    max_paths : int
        Max number of paths to plot per dimension
    dim : int
        Single dimension to plot (ignored if dims is provided)
    dims : list of int, optional
        List of dimensions to plot. If None, plots dimension `dim`.
    labels : list of str, optional
        Labels for each dimension

    Returns
    -------
    ax : Axes
    """
    ax = ax or plt.gca()
    n_paths, n_times, n_dims = paths.shape
    n_paths_to_plot = min(n_paths, max_paths)
    
    if dims is None:
        dims = [dim]
    
    if labels is None:
        labels = [f"dim {d}" for d in dims]
    
    for idx, d in enumerate(dims):
        if d >= n_dims:
            raise ValueError(f"dim {d} out of range for {n_dims}-dimensional paths")
        for i in range(n_paths_to_plot):
            label = labels[idx] if i == 0 else None
            ax.plot(times, paths[i, :, d], alpha=0.5, label=label)
    
    ax.set_xlabel("t")
    ax.set_ylabel("X_t")
    if len(dims) > 1 or labels[0] != "dim 0":
        ax.legend()
    return ax


def plot_mean_variance(
    times: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    ax: Optional[plt.Axes] = None,
    dim: int = 0,
    label: Optional[str] = None,
) -> plt.Axes:
    """Plot mean and variance bands for a single dimension.

    Parameters
    ----------
    times : ndarray
        Time points
    mean : ndarray
        Mean trajectory, shape (n_times, n_dims)
    var : ndarray
        Variance trajectory, shape (n_times, n_dims)
    ax : Axes, optional
        Matplotlib axes
    dim : int
        Dimension to plot
    label : str, optional
        Label for the mean line

    Returns
    -------
    ax : Axes
    """
    ax = ax or plt.gca()
    if dim >= mean.shape[1]:
        raise ValueError(f"dim {dim} out of range for {mean.shape[1]}-dimensional data")
    
    label = label or f"mean (dim {dim})"
    ax.plot(times, mean[:, dim], label=label)
    std = np.sqrt(var[:, dim])
    ax.fill_between(
        times,
        mean[:, dim] - std,
        mean[:, dim] + std,
        alpha=0.2,
        label=f"±1 std",
    )
    ax.set_xlabel("t")
    ax.legend()
    return ax
