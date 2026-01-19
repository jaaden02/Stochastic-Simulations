"""Plotting utilities for deterministic PDE solutions.

This submodule provides visualization functions for deterministic PDE solutions:

- evolution: Solution evolution and snapshot plots, spacetime heatmaps
- diagnostic_plots: Diagnostic time series (mass, norms, etc.)
- comparison: Numerical vs exact solution comparisons
"""
from .evolution import (
    plot_solution_evolution,
    plot_solution_snapshots,
    plot_spacetime_heatmap,
)
from .diagnostic_plots import plot_diagnostics
from .comparison import plot_comparison_with_exact

__all__ = [
    "plot_solution_evolution",
    "plot_solution_snapshots",
    "plot_spacetime_heatmap",
    "plot_diagnostics",
    "plot_comparison_with_exact",
]
