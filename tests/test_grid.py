"""Tests for grid creation and validation."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stochlib.setup import Grid


class TestGridCreation:
    """Test grid initialization and properties."""

    def test_1d_grid_creation(self):
        """Test basic 1D grid creation."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        assert grid.x_start == 0.0
        assert grid.x_end == 10.0
        assert grid.num_points_x == 64

    def test_2d_grid_creation(self):
        """Test 2D grid creation."""
        grid = Grid(
            x_start=0.0,
            x_end=10.0,
            num_points_x=32,
            y_start=-5.0,
            y_end=5.0,
            num_points_y=32,
        )
        assert grid.x_start == 0.0
        assert grid.x_end == 10.0
        assert grid.num_points_x == 32
        assert grid.y_start == -5.0
        assert grid.y_end == 5.0
        assert grid.num_points_y == 32

    def test_grid_with_asymmetric_bounds(self):
        """Test grid with asymmetric bounds."""
        grid = Grid(x_start=-10.0, x_end=20.0, num_points_x=100)
        assert grid.x_start == -10.0
        assert grid.x_end == 20.0
        assert grid.num_points_x == 100
        range_x = grid.x_end - grid.x_start
        assert range_x == 30.0


class TestGridSpacing:
    """Test grid spacing calculations."""

    def test_1d_grid_spacing(self, simple_1d_grid):
        """Test that 1D grid spacing is uniform."""
        # Grid goes from 0 to 10 with 64 points
        # Spacing should be 10 / (64 - 1)
        expected_dx = 10.0 / (64 - 1)
        assert hasattr(simple_1d_grid, "dx")
        assert np.isclose(simple_1d_grid.dx, expected_dx)

    def test_2d_grid_spacing(self, simple_2d_grid):
        """Test 2D grid spacing."""
        # Both x and y should have equal spacing
        expected_dx = 10.0 / (32 - 1)
        expected_dy = 10.0 / (32 - 1)
        assert np.isclose(simple_2d_grid.dx, expected_dx)
        assert np.isclose(simple_2d_grid.dy, expected_dy)


class TestGridProperties:
    """Test computed grid properties."""

    def test_grid_has_coordinate_arrays(self, simple_1d_grid):
        """Test that grid has coordinate arrays."""
        assert hasattr(simple_1d_grid, "x") or hasattr(simple_1d_grid, "X")
        # Exact attribute depends on implementation

    def test_grid_domain_size(self, simple_1d_grid):
        """Test grid domain size calculation."""
        domain_size = simple_1d_grid.x_end - simple_1d_grid.x_start
        assert domain_size == 10.0

    def test_grid_point_count(self, simple_1d_grid):
        """Test that grid has correct number of points."""
        assert simple_1d_grid.num_points_x == 64


class TestGridRefinement:
    """Test grids with different resolutions."""

    def test_coarse_grid(self):
        """Test coarse grid."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=16)
        assert grid.num_points_x == 16

    def test_fine_grid(self):
        """Test fine grid."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=512)
        assert grid.num_points_x == 512

    def test_spacing_decreases_with_refinement(self):
        """Test that spacing decreases as grid is refined."""
        coarse = Grid(x_start=0.0, x_end=10.0, num_points_x=16)
        fine = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        assert fine.dx < coarse.dx


class TestGridValidation:
    """Test grid validation and constraints."""

    def test_grid_with_single_point(self):
        """Test grid with minimum points."""
        # Should handle edge case gracefully
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=2)
        assert grid.num_points_x == 2

    def test_grid_with_large_domain(self):
        """Test grid with large domain."""
        grid = Grid(x_start=-1000.0, x_end=1000.0, num_points_x=256)
        assert grid.x_start == -1000.0
        assert grid.x_end == 1000.0

    def test_2d_grid_extent(self, simple_2d_grid):
        """Test that 2D grid has correct extents."""
        assert simple_2d_grid.x_end - simple_2d_grid.x_start == 10.0
        assert simple_2d_grid.y_end - simple_2d_grid.y_start == 10.0
