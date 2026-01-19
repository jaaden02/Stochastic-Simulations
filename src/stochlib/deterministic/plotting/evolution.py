"""Solution evolution visualization for deterministic PDEs."""

from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt


def plot_solution_evolution(
    t_eval: np.ndarray,
    u: np.ndarray,
    x: np.ndarray,
    ax: Optional[plt.Axes] = None,
    n_snapshots: int = 5,
    colormap: str = "Reds",
    alpha: float = 0.8,
) -> plt.Axes:
    """Plot solution evolution over time with multiple snapshots.

    Parameters
    ----------
    t_eval : ndarray
        Time points (n_times,)
    u : ndarray
        Solution array (n_times, n_grid)
    x : ndarray
        Spatial grid (n_grid,)
    ax : Axes, optional
        Matplotlib axes
    n_snapshots : int
        Number of time snapshots to plot
    colormap : str
        Colormap name
    alpha : float
        Line transparency

    Returns
    -------
    ax : Axes
    """
    ax = ax or plt.gca()

    n_times = u.shape[0]
    indices = np.linspace(0, n_times - 1, n_snapshots, dtype=int)

    cmap = plt.cm.get_cmap(colormap)

    for idx, t_idx in enumerate(indices):
        frac = idx / (n_snapshots - 1) if n_snapshots > 1 else 0
        color = cmap(0.3 + 0.6 * frac)
        ax.plot(
            x, u[t_idx], color=color, linewidth=2, alpha=alpha, label=f"t = {t_eval[t_idx]:.3f}"
        )

    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title("Solution Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_solution_snapshots(
    t_eval: np.ndarray,
    u: np.ndarray,
    x: np.ndarray,
    times: Optional[List[int]] = None,
    ax: Optional[plt.Axes] = None,
    colormap: str = "viridis",
) -> plt.Axes:
    """Plot solution at specific time indices.

    Parameters
    ----------
    t_eval : ndarray
        Time points
    u : ndarray
        Solution array (n_times, n_grid)
    x : ndarray
        Spatial grid
    times : list of int, optional
        Time indices to plot. If None, uses 4 evenly spaced snapshots.
    ax : Axes, optional
        Matplotlib axes
    colormap : str
        Colormap name

    Returns
    -------
    ax : Axes
    """
    ax = ax or plt.gca()

    n_times = u.shape[0]
    if times is None:
        times = np.linspace(0, n_times - 1, 4, dtype=int).tolist()
    else:
        times = [t for t in times if t < n_times]

    cmap = plt.cm.get_cmap(colormap)

    for idx, t_idx in enumerate(times):
        frac = idx / (len(times) - 1) if len(times) > 1 else 0
        ax.plot(x, u[t_idx], color=cmap(frac), linewidth=2, label=f"t = {t_eval[t_idx]:.3f}")

    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title("Solution Snapshots")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_spacetime_heatmap(
    t_eval: np.ndarray,
    u: np.ndarray,
    x: np.ndarray,
    ax: Optional[plt.Axes] = None,
    colormap: str = "RdYlBu_r",
) -> plt.Axes:
    """Plot solution as spacetime heatmap.

    Parameters
    ----------
    t_eval : ndarray
        Time points
    u : ndarray
        Solution array (n_times, n_grid)
    x : ndarray
        Spatial grid
    ax : Axes, optional
        Matplotlib axes
    colormap : str
        Colormap name

    Returns
    -------
    ax : Axes
    """
    ax = ax or plt.gca()

    T, X = np.meshgrid(t_eval, x, indexing="ij")

    im = ax.contourf(X, T, u, levels=50, cmap=colormap)
    plt.colorbar(im, ax=ax, label="u(x, t)")

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Spacetime Evolution")

    return ax
