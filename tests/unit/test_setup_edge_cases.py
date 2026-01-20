"""Comprehensive edge case tests for setup utilities and grid configuration."""

import pytest
import numpy as np
from stochlib import Grid, InitialCondition


class TestGrid1DEdgeCases:
    """Edge case tests for 1D grid creation and validation."""

    def test_grid_minimum_points(self):
        """Test grid with minimum valid points."""
        grid = Grid(num_points_x=2, x_start=0.0, x_end=1.0)
        assert grid.num_points_x == 2
        assert len(grid.x_grid) == 2

    def test_grid_maximum_points(self):
        """Test grid with very large number of points."""
        grid = Grid(num_points_x=10000, x_start=0.0, x_end=1.0)
        assert grid.num_points_x == 10000
        assert np.isfinite(grid.x_grid).all()

    def test_grid_spacing_uniformity(self):
        """Test that grid spacing is uniform."""
        grid = Grid(num_points_x=101, x_start=0.0, x_end=10.0)
        dx = np.diff(grid.x_grid)
        assert np.allclose(dx, grid.dx), "Grid spacing should be uniform"

    def test_grid_negative_domain(self):
        """Test grid with negative coordinate domain."""
        grid = Grid(num_points_x=51, x_start=-5.0, x_end=-1.0)
        assert grid.x_grid[0] == pytest.approx(-5.0)
        assert grid.x_grid[-1] == pytest.approx(-1.0)

    def test_grid_domain_spanning_zero(self):
        """Test grid that spans negative to positive."""
        grid = Grid(num_points_x=51, x_start=-2.5, x_end=2.5)
        assert grid.x_grid[0] == pytest.approx(-2.5)
        assert grid.x_grid[-1] == pytest.approx(2.5)
        # Check that zero is approximately in the grid
        assert np.any(np.abs(grid.x_grid) < grid.dx)

    def test_grid_very_small_domain(self):
        """Test grid with very small spatial extent."""
        grid = Grid(num_points_x=10, x_start=0.0, x_end=0.001)
        assert grid.x_grid[-1] - grid.x_grid[0] == pytest.approx(0.001)
        assert grid.dx == pytest.approx(0.001 / 9)

    def test_grid_very_large_domain(self):
        """Test grid with very large spatial extent."""
        grid = Grid(num_points_x=100, x_start=-1e6, x_end=1e6)
        assert np.isfinite(grid.x_grid).all()
        assert np.isfinite(grid.dx)


class TestGrid2DEdgeCases:
    """Edge case tests for 2D grid creation."""

    def test_grid_2d_square(self):
        """Test 2D grid with square domain."""
        grid = Grid(
            num_points_x=21, x_start=0.0, x_end=1.0, num_points_y=21, y_start=0.0, y_end=1.0
        )
        assert grid.shape == (21, 21)

    def test_grid_2d_rectangular(self):
        """Test 2D grid with rectangular domain."""
        grid = Grid(
            num_points_x=51, x_start=-1.0, x_end=1.0, num_points_y=21, y_start=-0.5, y_end=0.5
        )
        assert grid.shape == (51, 21)
        assert grid.dx != grid.dy  # Different aspect ratios

    def test_grid_2d_minimum_size(self):
        """Test 2D grid with minimum valid size."""
        grid = Grid(num_points_x=2, x_start=0.0, x_end=1.0, num_points_y=2, y_start=0.0, y_end=1.0)
        assert grid.shape == (2, 2)

    def test_grid_2d_very_different_aspect_ratio(self):
        """Test 2D grid with extreme aspect ratios."""
        grid = Grid(
            num_points_x=1000, x_start=0.0, x_end=100.0, num_points_y=2, y_start=0.0, y_end=1.0
        )
        assert grid.shape == (1000, 2)
        # y has much larger spacing than x due to fewer points
        assert grid.dy > grid.dx


class TestGrid3DEdgeCases:
    """Edge case tests for 3D grid creation."""

    def test_grid_3d_cubic(self):
        """Test 3D grid with cubic domain."""
        grid = Grid(
            num_points_x=16,
            x_start=0.0,
            x_end=1.0,
            num_points_y=16,
            y_start=0.0,
            y_end=1.0,
            num_points_z=16,
            z_start=0.0,
            z_end=1.0,
        )
        assert grid.shape == (16, 16, 16)
        assert np.isclose(grid.dx, grid.dy)
        assert np.isclose(grid.dy, grid.dz)

    def test_grid_3d_rectangular_box(self):
        """Test 3D grid with rectangular box domain."""
        grid = Grid(
            num_points_x=32,
            x_start=0.0,
            x_end=2.0,
            num_points_y=24,
            y_start=0.0,
            y_end=1.5,
            num_points_z=16,
            z_start=0.0,
            z_end=1.0,
        )
        assert grid.shape == (32, 24, 16)
        assert grid.dx < grid.dy < grid.dz  # Different spacings

    def test_grid_3d_minimum_size(self):
        """Test 3D grid with minimum valid size."""
        grid = Grid(
            num_points_x=2,
            x_start=0.0,
            x_end=1.0,
            num_points_y=2,
            y_start=0.0,
            y_end=1.0,
            num_points_z=2,
            z_start=0.0,
            z_end=1.0,
        )
        assert grid.shape == (2, 2, 2)
        assert grid.total_points == 8

    def test_grid_3d_asymmetric_domains(self):
        """Test 3D grid with asymmetric domain bounds."""
        grid = Grid(
            num_points_x=51,
            x_start=-1.0,
            x_end=1.0,
            num_points_y=31,
            y_start=-2.0,
            y_end=2.0,
            num_points_z=21,
            z_start=-0.5,
            z_end=0.5,
        )
        assert grid.shape == (51, 31, 21)
        assert grid.x_start == -1.0 and grid.x_end == 1.0
        assert grid.y_start == -2.0 and grid.y_end == 2.0
        assert grid.z_start == -0.5 and grid.z_end == 0.5

    def test_grid_3d_extreme_aspect_ratio(self):
        """Test 3D grid with extreme aspect ratios."""
        grid = Grid(
            num_points_x=256,
            x_start=0.0,
            x_end=1.0,
            num_points_y=16,
            y_start=0.0,
            y_end=1.0,
            num_points_z=2,
            z_start=0.0,
            z_end=1.0,
        )
        assert grid.shape == (256, 16, 2)
        # z has much larger spacing than x
        assert grid.dz > grid.dx

    def test_grid_3d_meshgrids_present(self):
        """Test that 3D grid has all coordinate meshgrids."""
        grid = Grid(
            num_points_x=8,
            x_start=0.0,
            x_end=1.0,
            num_points_y=8,
            y_start=0.0,
            y_end=1.0,
            num_points_z=8,
            z_start=0.0,
            z_end=1.0,
        )
        assert grid.X is not None
        assert grid.Y is not None
        assert grid.Z is not None
        assert grid.X.shape == (8, 8, 8)
        assert grid.Y.shape == (8, 8, 8)
        assert grid.Z.shape == (8, 8, 8)

    def test_grid_3d_large_volume(self):
        """Test 3D grid with moderately large volume."""
        grid = Grid(
            num_points_x=32,
            x_start=0.0,
            x_end=1.0,
            num_points_y=32,
            y_start=0.0,
            y_end=1.0,
            num_points_z=32,
            z_start=0.0,
            z_end=1.0,
        )
        assert grid.shape == (32, 32, 32)
        assert grid.total_points == 32**3
        # Grid should still be created (within memory limits)
        assert np.isfinite(grid.volume_element)


class TestInitialConditionEdgeCases:
    """Edge case tests for initial condition creation."""

    def test_ic_very_narrow_gaussian(self):
        """Test IC with very narrow Gaussian (small sigma)."""
        grid = Grid(num_points_x=1001, x_start=-10.0, x_end=10.0)
        ic = InitialCondition(grid, func_type="gaussian", x0=0.0, sigma_x=0.01)

        # Peak should be near center
        peak_idx = np.argmax(ic.f0)
        assert abs(grid.x_grid[peak_idx] - 0.0) < 0.1

        # Should be normalized
        integral = np.trapz(ic.f0, grid.x_grid)
        assert integral == pytest.approx(1.0, rel=1e-2)

    def test_ic_very_wide_gaussian(self):
        """Test IC with very wide Gaussian (large sigma)."""
        grid = Grid(num_points_x=101, x_start=-10.0, x_end=10.0)
        ic = InitialCondition(grid, func_type="gaussian", x0=0.0, sigma_x=5.0)

        # Should still be normalized
        integral = np.trapz(ic.f0, grid.x_grid)
        assert integral == pytest.approx(1.0, rel=1e-2)

    def test_ic_off_center_gaussian(self):
        """Test IC with off-center Gaussian."""
        grid = Grid(num_points_x=501, x_start=-10.0, x_end=10.0)
        center = 3.5
        ic = InitialCondition(grid, func_type="gaussian", x0=center, sigma_x=1.0)

        # Peak should be near specified center
        peak_idx = np.argmax(ic.f0)
        assert abs(grid.x_grid[peak_idx] - center) < grid.dx

    def test_ic_at_domain_boundary(self):
        """Test IC centered at domain boundary."""
        grid = Grid(num_points_x=101, x_start=0.0, x_end=10.0)
        ic = InitialCondition(grid, func_type="gaussian", x0=0.0, sigma_x=1.0)

        # Should still be normalized despite boundary
        integral = np.trapz(ic.f0, grid.x_grid)
        assert integral == pytest.approx(1.0, rel=1e-1)

    def test_ic_uniform_distribution(self):
        """Test uniform IC creation."""
        grid = Grid(num_points_x=51, x_start=0.0, x_end=1.0)
        ic = InitialCondition(grid, func_type="uniform")

        # All values should be equal
        assert np.allclose(ic.f0, ic.f0[0])

        # Should be normalized (trapz rule might have small error)
        integral = np.trapz(ic.f0, grid.x_grid)
        assert 0.98 < integral < 1.02  # Allow wider tolerance for discrete integration

    def test_ic_delta_approximation(self):
        """Test delta function approximation."""
        grid = Grid(num_points_x=1001, x_start=-10.0, x_end=10.0)
        center = 0.0
        ic = InitialCondition(grid, func_type="gaussian", x0=center, sigma_x=0.001)

        # Should be very sharp around center
        peak_idx = np.argmax(ic.f0)
        peak_value = ic.f0[peak_idx]

        # Values away from peak should be much smaller
        away_indices = np.where(np.abs(grid.x_grid - center) > 1.0)[0]
        assert np.all(ic.f0[away_indices] < peak_value * 0.01)


class TestGridValidationErrors:
    """Test that validation catches invalid configurations."""

    def test_grid_negative_points(self):
        """Test that negative point counts are rejected."""
        with pytest.raises(ValueError):
            Grid(num_points_x=-10, x_start=0.0, x_end=1.0)

    def test_grid_zero_points(self):
        """Test that zero points are rejected."""
        with pytest.raises(ValueError):
            Grid(num_points_x=0, x_start=0.0, x_end=1.0)

    def test_grid_equal_bounds(self):
        """Test that equal start/end bounds are rejected."""
        with pytest.raises(ValueError):
            Grid(num_points_x=51, x_start=1.0, x_end=1.0)

    def test_grid_inverted_bounds(self):
        """Test that inverted bounds are rejected."""
        with pytest.raises(ValueError):
            Grid(num_points_x=51, x_start=10.0, x_end=0.0)

    def test_grid_non_finite_bounds(self):
        """Test that non-finite bounds are rejected."""
        with pytest.raises(ValueError):
            Grid(num_points_x=51, x_start=0.0, x_end=np.inf)

        with pytest.raises(ValueError):
            Grid(num_points_x=51, x_start=np.nan, x_end=1.0)


class TestICValidationErrors:
    """Test that IC validation catches invalid configurations."""

    def test_ic_invalid_type(self):
        """Test that invalid IC types are rejected."""
        grid = Grid(num_points_x=51, x_start=0.0, x_end=1.0)

        with pytest.raises((ValueError, KeyError)):
            InitialCondition(grid, func_type="invalid_type")

    def test_ic_zero_sigma(self):
        """Test that zero sigma is handled."""
        grid = Grid(num_points_x=51, x_start=0.0, x_end=1.0)

        # Zero sigma might raise or create delta function
        try:
            ic = InitialCondition(grid, func_type="gaussian", sigma_x=0.0)
            # If it doesn't raise, it should still be normalized
            integral = np.trapz(ic.f0, grid.x_grid)
            assert integral > 0
        except ValueError:
            # Zero sigma properly rejected
            pass

    def test_ic_negative_sigma(self):
        """Test that negative sigma is rejected."""
        grid = Grid(num_points_x=51, x_start=0.0, x_end=1.0)

        with pytest.raises(ValueError):
            InitialCondition(grid, func_type="gaussian", sigma_x=-1.0)


class TestGridMemorySafety:
    """Test that grid handles memory constraints properly."""

    def test_grid_very_fine_resolution(self):
        """Test grid with very high resolution."""
        # Should not crash, but may have reasonable limits
        try:
            grid = Grid(num_points_x=100000, x_start=0.0, x_end=1.0)
            assert grid.num_points_x == 100000
        except (ValueError, MemoryError):
            # May reject if too large
            pass

    def test_grid_2d_large_total_points(self):
        """Test 2D grid doesn't exceed memory limits."""
        # 1000x1000 = 1M points should be fine
        try:
            grid = Grid(
                num_points_x=1000, x_start=0.0, x_end=1.0, num_points_y=1000, y_start=0.0, y_end=1.0
            )
            assert grid.shape == (1000, 1000)
        except (ValueError, MemoryError):
            # May reject if too large
            pass
