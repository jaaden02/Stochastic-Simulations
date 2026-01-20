"""Fokker-Planck PDE solver module.

Provides numerical solvers for Fokker-Planck equations, stability analysis,
regime selection, and diagnostic tools.
"""

from .solver import FokkerPlanckSolver
from .diagnostics import StabilityAnalyzer, DiagnosticReport, SolutionDiagnostics
from .selector import NumericalRegimeAdvisor, SimulationEngine
from .kernel import universal_fp_step_kernel
from . import plotting

__all__ = [
    "FokkerPlanckSolver",
    "StabilityAnalyzer",
    "DiagnosticReport",
    "SolutionDiagnostics",
    "NumericalRegimeAdvisor",
    "SimulationEngine",
    "universal_fp_step_kernel",
    "plotting",
]
