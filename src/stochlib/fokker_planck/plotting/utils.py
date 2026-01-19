"""High-level plotting interface and convenience wrappers."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from ...setup import Grid
from ...logging_utils import get_logger
from .distributions import (
    plot_1d_distribution,
    plot_1d_comparison,
    plot_1d_overlay,
    plot_2d_distribution,
)
from .diagnostics import (
    plot_mass_conservation,
    plot_entropy,
    plot_convergence,
    plot_solution_bounds,
    plot_diagnostics_summary,
)

logger = get_logger("plotting")


def plot_simulation_summary(
    grid: Grid,
    f_initial: np.ndarray,
    f_final: np.ndarray,
    history: Optional[Dict] = None,
    output_dir: str = ".",
    figname_prefix: str = "sim",
) -> Dict[str, str]:
    """Generate a complete set of summary plots for a simulation.

    Creates distribution comparison and (optionally) diagnostic plots.

    Parameters
    ----------
    grid : Grid
        Spatial grid configuration
    f_initial : ndarray
        Initial distribution
    f_final : ndarray
        Final distribution
    history : dict, optional
        Diagnostics history dictionary. If provided, generates diagnostic plots.
    output_dir : str
        Output directory for plots (default: current directory)
    figname_prefix : str
        Prefix for saved filenames (default: "sim")

    Returns
    -------
    dict
        Dictionary mapping plot descriptions to output filenames
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    output_files = {}

    # 1. Distribution comparison
    if len(grid.axis_names) == 1:
        # 1D plots
        path = os.path.join(output_dir, f"{figname_prefix}_initial_vs_final_overlay.png")
        plot_1d_overlay(grid, f_initial, f_final, output_path=path)
        output_files["1d_overlay"] = path

        path = os.path.join(output_dir, f"{figname_prefix}_initial_vs_final_sidebyside.png")
        plot_1d_comparison(grid, f_initial, f_final, output_path=path)
        output_files["1d_comparison"] = path
    else:
        # 2D plots
        path = os.path.join(output_dir, f"{figname_prefix}_initial.png")
        plot_2d_distribution(grid, f_initial, title="Initial Distribution", output_path=path)
        output_files["2d_initial"] = path

        path = os.path.join(output_dir, f"{figname_prefix}_final.png")
        plot_2d_distribution(grid, f_final, title="Final Distribution", output_path=path)
        output_files["2d_final"] = path

    # 2. Diagnostic plots (if history provided)
    if history is not None and history:
        path = os.path.join(output_dir, f"{figname_prefix}_diagnostics.png")
        fig = plot_diagnostics_summary(history, output_path=path)
        if fig is not None:
            output_files["diagnostics"] = path

        path = os.path.join(output_dir, f"{figname_prefix}_mass.png")
        fig = plot_mass_conservation(history, output_path=path)
        if fig is not None:
            output_files["mass"] = path

        path = os.path.join(output_dir, f"{figname_prefix}_entropy.png")
        fig = plot_entropy(history, output_path=path)
        if fig is not None:
            output_files["entropy"] = path

        path = os.path.join(output_dir, f"{figname_prefix}_convergence.png")
        fig = plot_convergence(history, output_path=path)
        if fig is not None:
            output_files["convergence"] = path

        path = os.path.join(output_dir, f"{figname_prefix}_bounds.png")
        fig = plot_solution_bounds(history, output_path=path)
        if fig is not None:
            output_files["bounds"] = path

    logger.info("Generated %d summary plots in %s/", len(output_files), output_dir)

    return output_files
