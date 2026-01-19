"""Tests for initial condition setup."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stochlib.setup import Grid, InitialCondition


class TestInitialConditionCreation:
    """Test IC initialization."""

    def test_gaussian_ic(self, simple_1d_grid):
        """Test Gaussian initial condition."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        assert ic.f0 is not None
        assert len(ic.f0) == simple_1d_grid.num_points_x

    def test_uniform_ic(self, simple_1d_grid):
        """Test uniform initial condition."""
        ic = InitialCondition(simple_1d_grid, func_type="uniform")
        assert ic.f0 is not None
        # Check that it's reasonably uniform
        assert np.std(ic.f0) < 0.1 * np.mean(ic.f0)

    def test_delta_ic(self, simple_1d_grid):
        """Test delta-like initial condition."""
        ic = InitialCondition(simple_1d_grid, func_type="delta", x0=5.0, sigma_x=0.1)
        assert ic.f0 is not None
        # Peak should be near x0
        peak_idx = np.argmax(ic.f0)
        assert 4.0 < simple_1d_grid.x_start + peak_idx * simple_1d_grid.dx < 6.0


class TestICNormalization:
    """Test IC normalization properties."""

    def test_gaussian_ic_normalized(self, simple_1d_grid):
        """Test that Gaussian IC integrates to 1."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        # Numerical integration with trapezoid rule
        integral = np.trapz(ic.f0, dx=simple_1d_grid.dx)
        assert np.isclose(integral, 1.0, rtol=0.01)

    def test_uniform_ic_normalized(self, simple_1d_grid):
        """Test that uniform IC integrates to 1."""
        ic = InitialCondition(simple_1d_grid, func_type="uniform")
        integral = np.trapz(ic.f0, dx=simple_1d_grid.dx)
        assert np.isclose(integral, 1.0, rtol=0.02)

    def test_delta_ic_normalized(self, simple_1d_grid):
        """Test that delta IC integrates to approximately 1."""
        ic = InitialCondition(simple_1d_grid, func_type="delta", x0=5.0, sigma_x=0.1)
        integral = np.trapz(ic.f0, dx=simple_1d_grid.dx)
        # Might be less than 1 due to discretization
        assert 0.8 < integral < 1.1


class TestICProperties:
    """Test IC mathematical properties."""

    def test_gaussian_ic_positivity(self, simple_1d_grid):
        """Test that Gaussian IC is non-negative."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        assert np.all(ic.f0 >= 0)

    def test_gaussian_ic_symmetry(self, simple_1d_grid):
        """Test Gaussian IC is roughly symmetric."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        # Mirror around center
        center_idx = simple_1d_grid.num_points_x // 2
        left = ic.f0[:center_idx]
        right = ic.f0[center_idx:][::-1]
        # Should be roughly equal
        assert np.corrcoef(left, right)[0, 1] > 0.95

    def test_delta_ic_peak_location(self, simple_1d_grid):
        """Test delta IC peak is at correct location."""
        x0 = 5.0
        ic = InitialCondition(simple_1d_grid, func_type="delta", x0=x0, sigma_x=0.2)
        peak_idx = np.argmax(ic.f0)
        peak_x = simple_1d_grid.x_start + peak_idx * simple_1d_grid.dx
        assert np.isclose(peak_x, x0, atol=0.5)


class TestICVariations:
    """Test IC with different parameters."""

    def test_narrow_gaussian(self, simple_1d_grid):
        """Test narrow Gaussian."""
        ic_narrow = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.1)
        ic_wide = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=1.0)
        # Narrow should have higher peak
        assert np.max(ic_narrow.f0) > np.max(ic_wide.f0)

    def test_centered_vs_off_center(self, simple_1d_grid):
        """Test centered vs off-center IC."""
        ic_center = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        ic_left = InitialCondition(simple_1d_grid, func_type="gaussian", x0=2.0, sigma_x=0.5)
        # Peaks should be at different positions
        center_peak_idx = np.argmax(ic_center.f0)
        left_peak_idx = np.argmax(ic_left.f0)
        assert center_peak_idx > left_peak_idx

    def test_2d_gaussian_ic(self):
        """Test 2D Gaussian IC."""
        grid = Grid(
            x_start=0.0,
            x_end=10.0,
            num_points_x=32,
            y_start=0.0,
            y_end=10.0,
            num_points_y=32,
        )
        ic = InitialCondition(grid, func_type="gaussian", x0=5.0, y0=5.0, sigma_x=0.5, sigma_y=0.5)
        assert ic.f0.ndim == 2
        assert ic.f0.shape == (32, 32)


class TestICDataTypes:
    """Test IC data type handling."""

    def test_ic_returns_float_array(self, simple_1d_grid):
        """Test that IC returns float array."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        assert isinstance(ic.f0, np.ndarray)
        assert np.issubdtype(ic.f0.dtype, np.floating)

    def test_ic_size_matches_grid(self, simple_1d_grid):
        """Test that IC size matches grid size."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        assert ic.f0.size == simple_1d_grid.num_points_x


class TestICEdgeCases:
    """Test IC edge cases."""

    def test_ic_at_boundary(self, simple_1d_grid):
        """Test IC centered at boundary."""
        ic_left = InitialCondition(simple_1d_grid, func_type="gaussian", x0=0.1, sigma_x=0.3)
        ic_right = InitialCondition(simple_1d_grid, func_type="gaussian", x0=9.9, sigma_x=0.3)
        assert ic_left.f0 is not None
        assert ic_right.f0 is not None

    def test_ic_with_very_small_sigma(self, simple_1d_grid):
        """Test IC with very small sigma."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.01)
        # Should still integrate reasonably
        integral = np.trapz(ic.f0, dx=simple_1d_grid.dx)
        assert 0.5 < integral < 1.5

    def test_ic_with_large_sigma(self, simple_1d_grid):
        """Test IC with large sigma."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=5.0)
        # Should still integrate reasonably
        integral = np.trapz(ic.f0, dx=simple_1d_grid.dx)
        assert 0.5 < integral < 1.5
