"""Results module for analyzing and comparing simulation outputs."""

from .result import SimulationResult
from .comparison import (
    ResultComparison,
    paths_to_histogram,
    compare_distributions,
    compare_moments,
)

__all__ = [
    "SimulationResult",
    "ResultComparison",
    "paths_to_histogram",
    "compare_distributions",
    "compare_moments",
]
