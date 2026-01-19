"""Deterministic PDE visualization for comparison with stochastic paths.

This module provides plotting functions for deterministic PDE solutions
and comparisons with stochastic path ensemble statistics.
"""
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_deterministic_solution(
    t_eval: np.ndarray,
    u: np.ndarray,
    x: np.ndarray,
    ax: Optional[plt.Axes] = None,
    label: str = "Deterministic",
    color: str = "red",
    alpha: float = 0.8,
) -> plt.Axes:
    """Plot deterministic PDE solution evolution.
    
    Parameters
    ----------
    t_eval : ndarray
        Time points (n_times,)
    u : ndarray
        Solution array (n_times, n_grid_points)
    x : ndarray
        Grid points (n_grid_points,)
    ax : Axes, optional
        Matplotlib axes
    label : str
        Label for the solution
    color : str
        Line color
    alpha : float
        Transparency
        
    Returns
    -------
    ax : Axes
    """
    ax = ax or plt.gca()
    
    # Plot solution at select times
    n_times = u.shape[0]
    times_to_plot = [0, n_times // 4, n_times // 2, 3 * n_times // 4, n_times - 1]
    times_to_plot = [t for t in times_to_plot if t < n_times]
    
    cmap = plt.cm.get_cmap('Reds')
    for idx, t_idx in enumerate(times_to_plot):
        frac = idx / len(times_to_plot)
        ax.plot(x, u[t_idx], color=cmap(0.3 + 0.6 * frac), 
                label=f"{label} (t={t_eval[t_idx]:.2f})", alpha=alpha)
    
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Deterministic PDE Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_deterministic_vs_stochastic(
    t_eval: np.ndarray,
    u_det: np.ndarray,
    x: np.ndarray,
    paths_mean: np.ndarray,
    paths_std: np.ndarray,
    ax: Optional[plt.Axes] = None,
    time_idx: int = -1,
) -> plt.Axes:
    """Plot deterministic solution vs stochastic ensemble statistics.
    
    Shows the deterministic solution alongside the mean and standard deviation
    of the stochastic paths at a specific time point.
    
    Parameters
    ----------
    t_eval : ndarray
        Time points
    u_det : ndarray
        Deterministic solution (n_times, n_grid_points)
    x : ndarray
        Grid points
    paths_mean : ndarray
        Mean of stochastic paths (n_grid_points,)
    paths_std : ndarray
        Standard deviation of stochastic paths (n_grid_points,)
    ax : Axes, optional
        Matplotlib axes
    time_idx : int
        Time index to plot (default: -1 for final time)
        
    Returns
    -------
    ax : Axes
    """
    ax = ax or plt.gca()
    
    # Plot deterministic solution at specified time
    ax.plot(x, u_det[time_idx], 'r-', linewidth=2, 
            label=f'Deterministic (t={t_eval[time_idx]:.2f})', zorder=5)
    
    # Plot stochastic mean and uncertainty
    ax.plot(x, paths_mean, 'b-', linewidth=2, label='Stochastic (mean)', zorder=5)
    ax.fill_between(x, paths_mean - paths_std, paths_mean + paths_std,
                    color='blue', alpha=0.2, label='±1 std dev')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.set_title('Deterministic vs Stochastic Solutions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_solution_snapshots(
    t_eval: np.ndarray,
    u: np.ndarray,
    x: np.ndarray,
    times: Optional[list] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot solution snapshots at specific times.
    
    Parameters
    ----------
    t_eval : ndarray
        Time points
    u : ndarray
        Solution array (n_times, n_grid_points)
    x : ndarray
        Grid points
    times : list, optional
        Time indices to plot. If None, plots 4 evenly spaced snapshots.
    ax : Axes, optional
        Matplotlib axes
        
    Returns
    -------
    ax : Axes
    """
    ax = ax or plt.gca()
    
    n_times = u.shape[0]
    if times is None:
        times = np.linspace(0, n_times - 1, 4, dtype=int)
    else:
        times = [t for t in times if t < n_times]
    
    cmap = plt.cm.get_cmap('viridis')
    for idx, t_idx in enumerate(times):
        frac = idx / (len(times) - 1) if len(times) > 1 else 0
        ax.plot(x, u[t_idx], color=cmap(frac), 
                label=f't={t_eval[t_idx]:.3f}', linewidth=2)
    
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Solution Snapshots')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
