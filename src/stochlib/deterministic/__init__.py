"""Deterministic PDE module for advection equations without diffusion.

This module provides solvers, diagnostics, and visualization tools for
deterministic advection PDEs of the form: ∂u/∂t + ∂(v·u)/∂x = 0
"""

from .solver import DeterministicPDESolver, solve_deterministic_pde
from .diagnostics import (
    DeterministicDiagnostics,
    check_mass_conservation,
    compute_cfl_number,
    compute_solution_moments,
    compute_l2_norm,
    compute_error_metrics,
    check_positivity,
    compute_total_variation,
)

__all__ = [
    # Solver
    "DeterministicPDESolver",
    "solve_deterministic_pde",
    # Diagnostics
    "DeterministicDiagnostics",
    "check_mass_conservation",
    "compute_cfl_number",
    "compute_solution_moments",
    "compute_l2_norm",
    "compute_error_metrics",
    "check_positivity",
    "compute_total_variation",
]
