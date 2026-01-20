"""Shared fixtures for all tests."""

import pytest
import sys
from pathlib import Path

# Add src to path so we can import stochlib
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import directly to avoid slow plotting module
from stochlib.setup import (
    Grid,
    InitialCondition,
    DiffusionConfig,
    VelocitiesConfig,
)
from stochlib.boundary_conditions import BoundaryConditions

# NOTE: SimulationEngine imports numba which is very slow
# Import it only in tests that need it


@pytest.fixture
def simple_1d_grid():
    """Simple 1D grid for testing."""
    return Grid(x_start=0.0, x_end=10.0, num_points_x=64)


@pytest.fixture
def fine_1d_grid():
    """Finer 1D grid for more accurate results."""
    return Grid(x_start=0.0, x_end=10.0, num_points_x=256)


@pytest.fixture
def medium_1d_grid():
    """Medium resolution 1D grid."""
    return Grid(x_start=0.0, x_end=10.0, num_points_x=128)


@pytest.fixture
def simple_2d_grid():
    """Simple 2D grid for testing."""
    return Grid(
        x_start=0.0,
        x_end=10.0,
        num_points_x=32,
        y_start=0.0,
        y_end=10.0,
        num_points_y=32,
    )


@pytest.fixture
def gaussian_ic_1d(simple_1d_grid):
    """1D Gaussian initial condition."""
    ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
    return ic.f0.copy()


@pytest.fixture
def uniform_ic_1d(simple_1d_grid):
    """1D uniform initial condition."""
    ic = InitialCondition(simple_1d_grid, func_type="uniform")
    return ic.f0.copy()


@pytest.fixture
def gaussian_ic_fine(fine_1d_grid):
    """Fine Gaussian initial condition."""
    ic = InitialCondition(fine_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
    return ic.f0.copy()


@pytest.fixture
def basic_setup_1d(simple_1d_grid):
    """Basic 1D setup: grid, IC, velocities, diffusion, BC."""
    ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.4)
    f0 = ic.f0.copy()

    velocities = VelocitiesConfig(simple_1d_grid, mu_x=-0.2)
    diffusions = DiffusionConfig(simple_1d_grid, constants={"x": 0.5})
    bc = BoundaryConditions(simple_1d_grid, bc_x="open")

    return {
        "grid": simple_1d_grid,
        "f0": f0,
        "velocities": velocities,
        "diffusions": diffusions,
        "bc": bc,
    }


# engine_1d fixture removed - import SimulationEngine only in tests that need it
# to avoid slow numba loading
