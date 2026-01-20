"""Tests for the results module."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stochlib.results import (
    SimulationResult,
    ResultComparison,
    paths_to_histogram,
    compare_distributions,
    compare_moments,
)
from stochlib.setup import Grid, InitialCondition, VelocitiesConfig, DiffusionConfig
from stochlib.boundary_conditions import BoundaryConditions


@pytest.fixture
def simple_1d_grid():
    """Create a simple 1D grid."""
    return Grid(x_start=0.0, x_end=10.0, num_points_x=64)


@pytest.fixture
def simple_1d_setup(simple_1d_grid):
    """Create a simple 1D setup."""
    grid = simple_1d_grid
    ic = InitialCondition(grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
    velocities = VelocitiesConfig(grid, mu_x=0.5)
    diffusions = DiffusionConfig(grid, constants={"x": 0.1})
    bc = BoundaryConditions(grid, bc_x="periodic")

    return {
        "grid": grid,
        "ic": ic,
        "velocities": velocities,
        "diffusions": diffusions,
        "bc": bc,
    }


class TestSimulationResult:
    """Test SimulationResult class."""

    def test_create_fp_result(self, simple_1d_setup):
        """Test creating a Fokker-Planck result."""
        setup = simple_1d_setup
        nt = 10
        t_array = np.linspace(0, 1, nt)
        data = np.random.rand(64, nt)  # (nx, nt)

        result = SimulationResult(
            solver_type="fokker_planck",
            data=data,
            t_array=t_array,
            grid=setup["grid"],
            initial_condition=setup["ic"],
            velocities=setup["velocities"],
            diffusions=setup["diffusions"],
            boundary_conditions=setup["bc"],
        )

        assert result.solver_type == "fokker_planck"
        assert result.is_fokker_planck()
        assert not result.is_sde()
        assert not result.is_deterministic()
        assert result.n_steps == nt

    def test_create_sde_result(self, simple_1d_setup):
        """Test creating an SDE result."""
        setup = simple_1d_setup
        nt = 10
        n_paths = 100
        t_array = np.linspace(0, 1, nt)
        data = np.random.rand(n_paths, nt, 1)  # (n_paths, nt, dim)

        result = SimulationResult(
            solver_type="sde",
            data=data,
            t_array=t_array,
            grid=setup["grid"],
            initial_condition=setup["ic"],
            velocities=setup["velocities"],
            diffusions=setup["diffusions"],
            boundary_conditions=setup["bc"],
        )

        assert result.is_sde()
        assert result.n_steps == nt

    def test_create_deterministic_result(self, simple_1d_setup):
        """Test creating a deterministic result."""
        setup = simple_1d_setup
        nt = 10
        n_traj = 50
        t_array = np.linspace(0, 1, nt)
        data = np.random.rand(n_traj, nt, 1)  # (n_trajectories, nt, dim)

        result = SimulationResult(
            solver_type="deterministic",
            data=data,
            t_array=t_array,
            grid=setup["grid"],
            initial_condition=setup["ic"],
            velocities=setup["velocities"],
            diffusions=setup["diffusions"],
            boundary_conditions=setup["bc"],
        )

        assert result.is_deterministic()

    def test_properties(self, simple_1d_setup):
        """Test result properties."""
        setup = simple_1d_setup
        nt = 10
        t_array = np.linspace(0, 2.0, nt)
        data = np.random.rand(64, nt)

        result = SimulationResult(
            solver_type="fokker_planck",
            data=data,
            t_array=t_array,
            grid=setup["grid"],
            initial_condition=setup["ic"],
            velocities=setup["velocities"],
            diffusions=setup["diffusions"],
            boundary_conditions=setup["bc"],
        )

        assert result.t_start == 0.0
        assert result.t_final == 2.0
        assert result.duration == 2.0
        assert result.n_steps == nt
        assert result.final.shape == (64,)
        assert result.initial.shape == (64,)

    def test_statistics_sde(self, simple_1d_setup):
        """Test ensemble statistics for SDE."""
        setup = simple_1d_setup
        nt = 5
        n_paths = 100
        t_array = np.linspace(0, 1, nt)
        # Create path data with known statistics
        data = np.random.randn(n_paths, nt, 1)  # (100, 5, 1)

        result = SimulationResult(
            solver_type="sde",
            data=data,
            t_array=t_array,
            grid=setup["grid"],
            initial_condition=setup["ic"],
            velocities=setup["velocities"],
            diffusions=setup["diffusions"],
            boundary_conditions=setup["bc"],
        )

        stats = result.statistics()

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["mean"].shape == (nt, 1)
        assert stats["std"].shape == (nt, 1)

    def test_statistics_raises_on_fp(self, simple_1d_setup):
        """Test that statistics() raises on FP result."""
        setup = simple_1d_setup
        nt = 10
        t_array = np.linspace(0, 1, nt)
        data = np.random.rand(64, nt)

        result = SimulationResult(
            solver_type="fokker_planck",
            data=data,
            t_array=t_array,
            grid=setup["grid"],
            initial_condition=setup["ic"],
            velocities=setup["velocities"],
            diffusions=setup["diffusions"],
            boundary_conditions=setup["bc"],
        )

        with pytest.raises(ValueError, match="Use moments"):
            result.statistics()

    def test_metadata(self, simple_1d_setup):
        """Test metadata handling."""
        setup = simple_1d_setup
        nt = 5
        t_array = np.linspace(0, 1, nt)
        data = np.random.rand(64, nt)
        custom_metadata = {"scheme": "upwind", "cfl": 0.5}

        result = SimulationResult(
            solver_type="fokker_planck",
            data=data,
            t_array=t_array,
            grid=setup["grid"],
            initial_condition=setup["ic"],
            velocities=setup["velocities"],
            diffusions=setup["diffusions"],
            boundary_conditions=setup["bc"],
            metadata=custom_metadata,
        )

        assert result.metadata["scheme"] == "upwind"
        assert result.metadata["cfl"] == 0.5
        assert "timestamp" in result.metadata
        assert result.metadata["solver_type"] == "fokker_planck"

    def test_to_dict(self, simple_1d_setup):
        """Test serialization to dict."""
        setup = simple_1d_setup
        nt = 3
        t_array = np.linspace(0, 1, nt)
        data = np.random.rand(64, nt)

        result = SimulationResult(
            solver_type="fokker_planck",
            data=data,
            t_array=t_array,
            grid=setup["grid"],
            initial_condition=setup["ic"],
            velocities=setup["velocities"],
            diffusions=setup["diffusions"],
            boundary_conditions=setup["bc"],
        )

        result_dict = result.to_dict()

        assert result_dict["solver_type"] == "fokker_planck"
        assert isinstance(result_dict["data"], list)
        assert isinstance(result_dict["t_array"], list)
        assert "metadata" in result_dict


class TestResultComparison:
    """Test ResultComparison class."""

    def test_create_comparison(self, simple_1d_setup):
        """Test creating a comparison between two results."""
        setup = simple_1d_setup
        nt = 5
        t_array = np.linspace(0, 1, nt)

        # Create two different results
        fp_data = np.random.rand(64, nt)
        fp_result = SimulationResult(
            solver_type="fokker_planck",
            data=fp_data,
            t_array=t_array,
            grid=setup["grid"],
            initial_condition=setup["ic"],
            velocities=setup["velocities"],
            diffusions=setup["diffusions"],
            boundary_conditions=setup["bc"],
        )

        sde_data = np.random.rand(100, nt, 1)
        sde_result = SimulationResult(
            solver_type="sde",
            data=sde_data,
            t_array=t_array,
            grid=setup["grid"],
            initial_condition=setup["ic"],
            velocities=setup["velocities"],
            diffusions=setup["diffusions"],
            boundary_conditions=setup["bc"],
        )

        comparison = ResultComparison([fp_result, sde_result])

        assert len(comparison.results) == 2
        assert comparison.labels == ["fokker_planck", "sde"]

    def test_comparison_custom_labels(self, simple_1d_setup):
        """Test comparison with custom labels."""
        setup = simple_1d_setup
        nt = 3
        t_array = np.linspace(0, 1, nt)
        data = np.random.rand(64, nt)

        result1 = SimulationResult(
            solver_type="fokker_planck",
            data=data,
            t_array=t_array,
            grid=setup["grid"],
            initial_condition=setup["ic"],
            velocities=setup["velocities"],
            diffusions=setup["diffusions"],
            boundary_conditions=setup["bc"],
        )

        result2 = SimulationResult(
            solver_type="fokker_planck",
            data=data * 0.9,
            t_array=t_array,
            grid=setup["grid"],
            initial_condition=setup["ic"],
            velocities=setup["velocities"],
            diffusions=setup["diffusions"],
            boundary_conditions=setup["bc"],
        )

        comparison = ResultComparison([result1, result2], labels=["fine_grid", "coarse_grid"])

        assert comparison.labels == ["fine_grid", "coarse_grid"]

    def test_comparison_requires_multiple_results(self, simple_1d_setup):
        """Test that comparison requires at least 2 results."""
        setup = simple_1d_setup
        nt = 3
        t_array = np.linspace(0, 1, nt)
        data = np.random.rand(64, nt)

        result = SimulationResult(
            solver_type="fokker_planck",
            data=data,
            t_array=t_array,
            grid=setup["grid"],
            initial_condition=setup["ic"],
            velocities=setup["velocities"],
            diffusions=setup["diffusions"],
            boundary_conditions=setup["bc"],
        )

        with pytest.raises(ValueError, match="at least 2"):
            ResultComparison([result])

    def test_summary_table(self, simple_1d_setup):
        """Test summary table generation."""
        setup = simple_1d_setup
        nt = 5
        t_array = np.linspace(0, 2.0, nt)
        data = np.random.rand(64, nt)

        result1 = SimulationResult(
            solver_type="fokker_planck",
            data=data,
            t_array=t_array,
            grid=setup["grid"],
            initial_condition=setup["ic"],
            velocities=setup["velocities"],
            diffusions=setup["diffusions"],
            boundary_conditions=setup["bc"],
        )

        result2 = SimulationResult(
            solver_type="sde",
            data=np.random.rand(100, nt, 1),
            t_array=t_array,
            grid=setup["grid"],
            initial_condition=setup["ic"],
            velocities=setup["velocities"],
            diffusions=setup["diffusions"],
            boundary_conditions=setup["bc"],
        )

        comparison = ResultComparison([result1, result2])
        table = comparison.summary_table()

        assert "fokker_planck" in table
        assert "sde" in table
        assert table["fokker_planck"]["t_final"] == 2.0
        assert table["fokker_planck"]["duration"] == 2.0
        assert table["fokker_planck"]["n_steps"] == nt


class TestComparisonUtilities:
    """Test comparison utility functions."""

    def test_paths_to_histogram(self, simple_1d_grid):
        """Test converting paths to histogram."""
        grid = simple_1d_grid
        n_paths = 100
        nt = 5

        # Create paths concentrated at x=5
        paths = np.random.normal(5.0, 0.1, (n_paths, nt, 1))

        hist = paths_to_histogram(paths, grid, time_idx=-1)

        assert hist.shape == grid.shape
        assert np.isclose(np.sum(hist) * grid.volume_element, 1.0, atol=1e-6)
        # Peak should be near x=5
        peak_idx = np.argmax(hist)
        assert grid.x_grid[peak_idx] > 4.5
        assert grid.x_grid[peak_idx] < 5.5

    def test_compare_distributions_identical(self, simple_1d_grid):
        """Test comparing identical distributions."""
        grid = simple_1d_grid
        f = np.exp(-((grid.x_grid - 5.0) ** 2) / 2)
        f = f / (np.sum(f) * grid.volume_element)

        metrics = compare_distributions(f, f, grid)

        assert np.isclose(metrics["l1_error"], 0.0, atol=1e-10)
        assert np.isclose(metrics["l2_error"], 0.0, atol=1e-10)
        assert np.isclose(metrics["kl_divergence"], 0.0, atol=1e-10)

    def test_compare_distributions_shifted(self, simple_1d_grid):
        """Test comparing shifted distributions."""
        grid = simple_1d_grid
        f1 = np.exp(-((grid.x_grid - 5.0) ** 2) / 2)
        f1 = f1 / (np.sum(f1) * grid.volume_element)

        f2 = np.exp(-((grid.x_grid - 5.5) ** 2) / 2)
        f2 = f2 / (np.sum(f2) * grid.volume_element)

        metrics = compare_distributions(f1, f2, grid)

        # Should be non-zero errors
        assert metrics["l1_error"] > 0
        assert metrics["l2_error"] > 0
        assert metrics["kl_divergence"] > 0

    def test_compare_moments(self, simple_1d_grid):
        """Test moment comparison."""
        grid = simple_1d_grid

        # Create a FP distribution
        f_fp = np.exp(-((grid.x_grid - 5.0) ** 2) / 2)
        f_fp = f_fp / (np.sum(f_fp) * grid.volume_element)

        # Fake path statistics
        mean_paths = np.array([5.1])
        var_paths = np.array([0.95])

        result = compare_moments(f_fp, grid, mean_paths, var_paths)

        assert "mean_fp" in result
        assert "var_fp" in result
        assert "mean_error" in result
        assert "var_error" in result
        assert result["mean_fp"].shape == (1,)
        assert result["var_fp"].shape == (1,)
