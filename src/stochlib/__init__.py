"""StochLib: Stochastic and deterministic PDE simulation library.

This package provides tools for solving Fokker-Planck PDEs, stochastic differential
equations (SDEs), and deterministic advection PDEs. It includes numerical solvers,
diagnostic tools, and visualization capabilities.
"""

from .setup import Grid, InitialCondition, DiffusionConfig, VelocitiesConfig, SimulationSetup
from .boundary_conditions import BoundaryConditions
from .logging_utils import configure_logging, get_logger, log_performance
from .debug_utils import (
    enable_debug,
    disable_debug,
    is_debug_mode,
    require_debug_mode,
    assert_finite,
    assert_positive,
    assert_shape,
    dump_state,
    inspect_array,
    print_array_info,
    profile_function,
    breakpoint_on_condition,
    breakpoint_if_nan,
    breakpoint_if_inf,
    debug_context,
)
from .sde import PathSimulator, StepSchemeAdvisor, PathDiagnostics
from .deterministic import DeterministicPDESolver, solve_deterministic_pde
from .results import (
    SimulationResult,
    ResultComparison,
    paths_to_histogram,
    compare_distributions,
    compare_moments,
)

__all__ = [
    "Grid",
    "InitialCondition",
    "DiffusionConfig",
    "VelocitiesConfig",
    "SimulationSetup",
    "BoundaryConditions",
    "configure_logging",
    "get_logger",
    "log_performance",
    "enable_debug",
    "disable_debug",
    "is_debug_mode",
    "require_debug_mode",
    "assert_finite",
    "assert_positive",
    "assert_shape",
    "dump_state",
    "inspect_array",
    "print_array_info",
    "profile_function",
    "breakpoint_on_condition",
    "breakpoint_if_nan",
    "breakpoint_if_inf",
    "debug_context",
    "PathSimulator",
    "StepSchemeAdvisor",
    "PathDiagnostics",
    "DeterministicPDESolver",
    "solve_deterministic_pde",
    "SimulationResult",
    "ResultComparison",
    "paths_to_histogram",
    "compare_distributions",
    "compare_moments",
]


def hello() -> str:
    return "Hello from stochastic-simulations!"
