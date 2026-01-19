"""Individual path visualization for SDE paths with threshold-based sampling."""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_paths_with_threshold(
    paths: np.ndarray,
    times: np.ndarray,
    alive_mask: Optional[np.ndarray] = None,
    dim: int = 0,
    max_paths_to_plot: int = 1000,
    rng: Optional[np.random.Generator] = None,
    figs_axes: Optional[tuple] = None,
    title_suffix: str = "",
) -> tuple:
    """Plot individual paths with threshold check to avoid matplotlib overload.

    When n_paths > max_paths_to_plot, randomly samples paths to display with a warning.

    Parameters
    ----------
    paths : ndarray
        Shape (n_paths, n_times, n_dims)
    times : ndarray
        Time points (n_times,)
    alive_mask : ndarray, optional
        Boolean mask (n_paths, n_times) indicating active paths per time step.
        If provided, plots only the alive portion of each path.
    dim : int
        Dimension to plot (default 0 for 1D)
    max_paths_to_plot : int
        Threshold: if n_paths > this, sample and warn (default 1000)
    rng : Generator, optional
        Random number generator for sampling paths (default: new Generator)
    figs_axes : tuple, optional
        (fig, axes) to use existing figure. If None, creates new figure.
    title_suffix : str
        Suffix for plot title (e.g., "Absorb" or "Reflect")

    Returns
    -------
    (fig, axes) : tuple of (Figure, Axes)
        The matplotlib figure and axes
    """
    n_paths = paths.shape[0]
    rng = rng or np.random.default_rng()

    # Determine which paths to plot
    if n_paths > max_paths_to_plot:
        indices = rng.choice(n_paths, size=max_paths_to_plot, replace=False)
        n_plot = max_paths_to_plot
        print(f"⚠ Plotting {n_plot}/{n_paths} sampled paths (threshold: {max_paths_to_plot})")
    else:
        indices = np.arange(n_paths)
        n_plot = n_paths
        print(f"Plotting all {n_plot} paths")

    # Create or use existing axes
    if figs_axes is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig, ax = figs_axes

    # Plot selected paths
    if alive_mask is not None:
        # Plot only alive portions (respects boundary absorption)
        for i in indices:
            mask_alive = alive_mask[i]
            ax.plot(times[mask_alive], paths[i, mask_alive, dim], alpha=0.1, color="blue")
    else:
        # Plot full paths
        for i in indices:
            ax.plot(times, paths[i, :, dim], alpha=0.1, color="blue")

    # Formatting
    ax.set_xlabel("t")
    ax.set_ylabel("Position")
    ax.set_title(f'Paths ({n_plot}/{n_paths} shown){" - " + title_suffix if title_suffix else ""}')
    ax.grid(True, alpha=0.3)

    return fig, ax
