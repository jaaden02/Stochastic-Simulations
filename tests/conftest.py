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


# Code Quality Checks
# These run automatically when pytest runs


def pytest_configure(config):
    """Run code quality checks before tests."""
    import subprocess

    # Only run in normal test mode, not in collection-only mode
    if config.option.collectonly:
        return

    root = Path(__file__).parent.parent

    # Run Black check
    print("\n" + "=" * 70)
    print("Running Black formatting check...")
    print("=" * 70)
    result = subprocess.run(
        ["uv", "run", "black", "--check", "--line-length", "100", "src/", "tests/"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise pytest.exit(
            "Black formatting check failed. Run: uv run black --line-length 100 src/ tests/", 1
        )
    print("[OK] Black check passed")

    # Run Ruff check
    print("\n" + "=" * 70)
    print("Running Ruff linting check...")
    print("=" * 70)
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "src/", "tests/"], cwd=root, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise pytest.exit("Ruff linting check failed. Run: uv run ruff check --fix src/ tests/", 1)
    print("[OK] Ruff check passed")

    # Run mypy type check
    print("\n" + "=" * 70)
    print("Running mypy type checking...")
    print("=" * 70)
    result = subprocess.run(
        ["uv", "run", "mypy", "src/stochlib", "--ignore-missing-imports"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print("\nNote: mypy warnings detected (non-fatal, see above)")
    else:
        print("[OK] mypy check passed")

    # Run profiling examples
    print("\n" + "=" * 70)
    print("Running profiling examples...")
    print("=" * 70)
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        ["uv", "run", "python", "profiling/profiling_examples.py"],
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        raise pytest.exit("Profiling examples failed", 1)
    print(result.stdout)
    print("[OK] Profiling examples completed")

    print("\n" + "=" * 70)
    print("All quality checks passed! Running tests...")
    print("=" * 70 + "\n")
