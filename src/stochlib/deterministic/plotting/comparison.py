"""Comparison plots for deterministic solutions."""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_comparison_with_exact(
    x: np.ndarray,
    u_numeric: np.ndarray,
    u_exact: np.ndarray,
    t: float,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot numerical solution compared to exact solution.

    Parameters
    ----------
    x : ndarray
        Spatial grid
    u_numeric : ndarray
        Numerical solution (n_grid,)
    u_exact : ndarray
        Exact solution (n_grid,)
    t : float
        Time value
    ax : Axes, optional
        Matplotlib axes

    Returns
    -------
    ax : Axes
    """
    ax = ax or plt.gca()

    ax.plot(x, u_exact, "k-", linewidth=2, label="Exact", zorder=5)
    ax.plot(x, u_numeric, "r--", linewidth=2, label="Numerical", zorder=4)

    # Plot error (scaled for visibility)
    error = u_numeric - u_exact
    ax.fill_between(x, 0, error * 10, alpha=0.3, color="orange", label="Error (×10)")

    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title(f"Numerical vs Exact Solution (t = {t:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
