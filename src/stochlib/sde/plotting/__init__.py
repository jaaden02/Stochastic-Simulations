"""Plotting helpers for SDE paths and deterministic PDEs.

This submodule provides visualization functions for stochastic differential equation
path simulations and deterministic PDE solutions. Functions are organized by type:

- trajectories: Individual trajectory plots and mean ± variance bands
- phase_portrait: 2D phase space visualizations
- paths: Individual path plots with threshold-based sampling for large ensembles
- deterministic: Deterministic PDE solution visualization and stochastic comparisons
"""
from .trajectories import plot_trajectories, plot_mean_variance
from .phase_portrait import plot_phase_portrait
from .paths import plot_paths_with_threshold
from .deterministic import (
    plot_deterministic_solution,
    plot_deterministic_vs_stochastic,
    plot_solution_snapshots,
)

__all__ = [
    "plot_trajectories",
    "plot_mean_variance",
    "plot_phase_portrait",
    "plot_paths_with_threshold",
    "plot_deterministic_solution",
    "plot_deterministic_vs_stochastic",
    "plot_solution_snapshots",
]
