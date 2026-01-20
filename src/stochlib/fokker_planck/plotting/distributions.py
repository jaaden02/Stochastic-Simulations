"""Plotting functions for probability distributions across dimensions."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from ...setup import Grid
from ...logging_utils import get_logger

logger = get_logger("plotting.distributions")


def plot_1d_distribution(
    grid: Grid,
    f: np.ndarray,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    linestyle: str = "-",
    linewidth: float = 2.0,
    color: Optional[str] = None,
    **kwargs,
) -> plt.Axes:
    """Plot a 1D probability distribution.

    Parameters
    ----------
    grid : Grid
        Spatial grid configuration
    f : ndarray
        1D probability distribution array (shape: N_x,)
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, uses current axes.
    label : str, optional
        Legend label
    linestyle : str
        Line style (default: '-')
    linewidth : float
        Line width (default: 2.0)
    color : str, optional
        Line color
    **kwargs
        Additional arguments passed to ax.plot()

    Returns
    -------
    plt.Axes
        The axes object with the plot
    """
    if ax is None:
        ax = plt.gca()

    if "x" not in grid.axis_names:
        raise ValueError("Grid must have x-axis for 1D plotting")

    ax.plot(
        grid.x_grid, f, linestyle=linestyle, linewidth=linewidth, color=color, label=label, **kwargs
    )
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    return ax


def plot_1d_comparison(
    grid: Grid,
    f_initial: np.ndarray,
    f_final: np.ndarray,
    figsize: Tuple[float, float] = (10, 5),
    title: str = "1D Distribution: Initial vs Final",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot initial and final 1D distributions side-by-side.

    Parameters
    ----------
    grid : Grid
        Spatial grid configuration
    f_initial : ndarray
        Initial distribution (shape: N_x,)
    f_final : ndarray
        Final distribution (shape: N_x,)
    figsize : tuple
        Figure size (width, height)
    title : str
        Figure title
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Initial distribution
    plot_1d_distribution(grid, f_initial, ax=ax1, label="initial", color="C0")
    ax1.set_title("Initial Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Final distribution
    plot_1d_distribution(grid, f_final, ax=ax2, label="final", color="C1")
    ax2.set_title("Final Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info("Saved 1D comparison plot to: %s", output_path)

    return fig


def plot_1d_overlay(
    grid: Grid,
    f_initial: np.ndarray,
    f_final: np.ndarray,
    figsize: Tuple[float, float] = (8, 5),
    title: str = "1D Distribution: Initial vs Final (Overlay)",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot initial and final 1D distributions overlaid on the same axes.

    Parameters
    ----------
    grid : Grid
        Spatial grid configuration
    f_initial : ndarray
        Initial distribution (shape: N_x,)
    f_final : ndarray
        Final distribution (shape: N_x,)
    figsize : tuple
        Figure size (width, height)
    title : str
        Figure title
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    plot_1d_distribution(grid, f_initial, ax=ax, label="initial", linestyle="--", linewidth=2.0)
    plot_1d_distribution(grid, f_final, ax=ax, label="final", linestyle="-", linewidth=2.5)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info("Saved 1D overlay plot to: %s", output_path)

    return fig


def plot_2d_distribution(
    grid: Grid,
    f: np.ndarray,
    slice_z: Optional[int] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    cmap: str = "viridis",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a 2D slice of the probability distribution.

    Parameters
    ----------
    grid : Grid
        Spatial grid configuration
    f : ndarray
        Distribution array (shape: N_x, N_z)
    slice_z : int, optional
        Index of z-slice (if 2D grid). If None, uses middle slice.
    figsize : tuple
        Figure size (width, height)
    title : str, optional
        Figure title
    cmap : str
        Colormap name (default: 'viridis')
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Handle 1D vs 2D grids
    if len(f.shape) == 1:
        raise ValueError("Use plot_1d_distribution for 1D data")
    elif len(f.shape) == 2:
        data = f
        xlabel = grid.axis_names[0] if len(grid.axis_names) > 0 else "x"
        ylabel = grid.axis_names[1] if len(grid.axis_names) > 1 else "y"
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {f.shape}")

    im = ax.imshow(data.T, origin="lower", cmap=cmap, aspect="auto")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, label="f")

    if title is None:
        title = "2D Distribution"
    ax.set_title(title, fontweight="bold")

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info("Saved 2D distribution plot to: %s", output_path)

    return fig
