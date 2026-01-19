"""StochLib: Stochastic and deterministic PDE simulation library.

This package provides tools for solving Fokker-Planck PDEs, stochastic differential
equations (SDEs), and deterministic advection PDEs. It includes numerical solvers,
diagnostic tools, and visualization capabilities.
"""
from .setup import Grid, InitialCondition, DiffusionConfig, VelocitiesConfig, SimulationSetup
from .boundary_conditions import BoundaryConditions
from .logging_utils import configure_logging, get_logger, log_performance
from .sde import PathSimulator, StepSchemeAdvisor, PathDiagnostics
from .deterministic import DeterministicPDESolver, solve_deterministic_pde

__all__ = [
    'Grid',
    'InitialCondition',
    'DiffusionConfig',
    'VelocitiesConfig',
    'SimulationSetup',
    'BoundaryConditions',
    'configure_logging',
    'get_logger',
    'log_performance',
    'PathSimulator',
    'StepSchemeAdvisor',
    'PathDiagnostics',
    'DeterministicPDESolver',
    'solve_deterministic_pde',
]

def hello() -> str:
    return "Hello from stochastic-simulations!"
