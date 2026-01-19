"""Plotting module for probability distributions and diagnostics."""

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

from .utils import plot_simulation_summary

__all__ = [
    # Distribution plots
    'plot_1d_distribution',
    'plot_1d_comparison',
    'plot_1d_overlay',
    'plot_2d_distribution',
    # Diagnostic plots
    'plot_mass_conservation',
    'plot_entropy',
    'plot_convergence',
    'plot_solution_bounds',
    'plot_diagnostics_summary',
    # High-level utils
    'plot_simulation_summary',
]
