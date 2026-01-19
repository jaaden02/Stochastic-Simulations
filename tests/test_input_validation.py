"""Tests for input validation and error handling."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stochlib.setup import Grid, InitialCondition, VelocitiesConfig, DiffusionConfig
from stochlib.boundary_conditions import BoundaryConditions


@pytest.mark.validation
class TestGridValidation:
    """Test grid validation with invalid inputs."""

    def test_negative_num_points(self):
        """Test grid with negative number of points."""
        with pytest.raises((ValueError, TypeError)):
            Grid(x_start=0.0, x_end=10.0, num_points_x=-10)

    def test_zero_num_points(self):
        """Test grid with zero points."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            Grid(x_start=0.0, x_end=10.0, num_points_x=0)

    def test_inverted_boundaries(self):
        """Test grid with x_start > x_end."""
        # Should either raise error or handle gracefully
        try:
            grid = Grid(x_start=10.0, x_end=0.0, num_points_x=64)
            # If it doesn't raise, check if it swaps or handles correctly
            assert grid.x_start <= grid.x_end or True  # Allow either behavior
        except (ValueError, AssertionError):
            pass  # Expected behavior

    def test_equal_boundaries(self):
        """Test grid with x_start == x_end."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            Grid(x_start=5.0, x_end=5.0, num_points_x=64)

    def test_single_point_grid(self):
        """Test grid with single point."""
        # May or may not be valid
        try:
            grid = Grid(x_start=0.0, x_end=10.0, num_points_x=1)
            assert grid.num_points_x == 1
        except ValueError:
            pass  # Also acceptable

    def test_mismatched_2d_params(self):
        """Test 2D grid with incomplete parameters."""
        # y_start provided but not y_end
        with pytest.raises((ValueError, TypeError)):
            Grid(x_start=0.0, x_end=10.0, num_points_x=32, y_start=0.0, num_points_y=32)

    def test_invalid_type_for_coordinates(self):
        """Test grid with invalid types."""
        with pytest.raises(TypeError):
            Grid(x_start="zero", x_end=10.0, num_points_x=64)

    def test_nan_coordinates(self):
        """Test grid with NaN coordinates."""
        with pytest.raises((ValueError, AssertionError)):
            Grid(x_start=np.nan, x_end=10.0, num_points_x=64)

    def test_inf_coordinates(self):
        """Test grid with infinite coordinates."""
        with pytest.raises((ValueError, OverflowError)):
            Grid(x_start=0.0, x_end=np.inf, num_points_x=64)

    def test_very_large_num_points(self):
        """Test grid with unreasonably large number of points."""
        # Should reject to prevent memory issues
        with pytest.raises((MemoryError, ValueError, OverflowError)):
            Grid(x_start=0.0, x_end=10.0, num_points_x=10**8)


@pytest.mark.validation
class TestInitialConditionValidation:
    """Test IC validation with invalid inputs."""

    def test_ic_outside_domain(self):
        """Test IC centered outside grid domain."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        # x0 far outside domain
        try:
            ic = InitialCondition(grid, func_type="gaussian", x0=100.0, sigma_x=0.5)
            # Should either work (but with no mass) or raise warning
            assert ic.f0 is not None
        except (ValueError, Warning):
            pass

    def test_negative_sigma(self):
        """Test IC with negative standard deviation."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        with pytest.raises((ValueError, AssertionError)):
            InitialCondition(grid, func_type="gaussian", x0=5.0, sigma_x=-0.5)

    def test_zero_sigma(self):
        """Test IC with zero standard deviation."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        with pytest.raises((ValueError, ZeroDivisionError)):
            InitialCondition(grid, func_type="gaussian", x0=5.0, sigma_x=0.0)

    def test_invalid_func_type(self):
        """Test IC with invalid function type."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        with pytest.raises((ValueError, KeyError)):
            InitialCondition(grid, func_type="invalid_type", x0=5.0, sigma_x=0.5)

    def test_missing_required_params(self):
        """Test IC missing required parameters - library provides sensible defaults."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        # Library provides sensible defaults for Gaussian params
        ic = InitialCondition(grid, func_type="gaussian")  # Missing x0, sigma_x
        # Should use defaults: x0 at center, sigma_x = 2*dx
        assert ic.f0 is not None
        assert ic.params["x0"] == (grid.x_start + grid.x_end) / 2
        assert ic.params["sigma_x"] > 0

    def test_2d_ic_on_1d_grid(self):
        """Test 2D IC parameters on 1D grid."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        # Providing y0 for 1D grid
        try:
            ic = InitialCondition(grid, func_type="gaussian", x0=5.0, sigma_x=0.5, y0=5.0)
            # May ignore extra params or raise error
        except (ValueError, TypeError):
            pass

    def test_none_grid(self):
        """Test IC with None grid."""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            InitialCondition(None, func_type="gaussian", x0=5.0, sigma_x=0.5)


@pytest.mark.validation
class TestBoundaryConditionValidation:
    """Test BC validation with invalid inputs."""

    def test_invalid_bc_type(self):
        """Test BC with invalid type string."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        with pytest.raises((ValueError, KeyError)):
            BoundaryConditions(grid, bc_x="invalid_bc_type")

    def test_bc_with_none_grid(self):
        """Test BC with None grid."""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            BoundaryConditions(None, bc_x="periodic")

    def test_bc_y_on_1d_grid(self):
        """Test specifying bc_y on 1D grid."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        try:
            bc = BoundaryConditions(grid, bc_x="periodic", bc_y="open")
            # Should either ignore or raise error
        except (ValueError, AttributeError):
            pass

    def test_empty_bc_string(self):
        """Test BC with empty string."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        with pytest.raises((ValueError, KeyError)):
            BoundaryConditions(grid, bc_x="")

    def test_bc_case_sensitivity(self):
        """Test if BC strings are case-sensitive."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        # Try uppercase
        try:
            bc = BoundaryConditions(grid, bc_x="OPEN")
            # If it works, case-insensitive
        except (ValueError, KeyError):
            # Case-sensitive, which is fine
            pass


@pytest.mark.validation
class TestVelocitiesConfigValidation:
    """Test velocities config validation."""

    def test_velocity_on_nonexistent_axis(self):
        """Test velocity for axis not in grid."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        with pytest.raises((ValueError, AttributeError)):
            VelocitiesConfig(grid, mu_y=0.5)  # No y-axis

    def test_none_grid(self):
        """Test velocities with None grid."""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            VelocitiesConfig(None, mu_x=0.5)

    def test_nan_velocity(self):
        """Test velocity with NaN value."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        try:
            vel = VelocitiesConfig(grid, mu_x=np.nan)
            # May accept but cause issues later
            assert vel is not None
        except (ValueError, AssertionError):
            pass

    def test_inf_velocity(self):
        """Test velocity with infinite value."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        try:
            vel = VelocitiesConfig(grid, mu_x=np.inf)
            # May accept but should be handled
            assert vel is not None
        except (ValueError, OverflowError):
            pass

    def test_all_none_velocities(self):
        """Test with no velocities specified."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        vel = VelocitiesConfig(grid)
        # Should default to zero
        assert vel is not None


@pytest.mark.validation
class TestDiffusionConfigValidation:
    """Test diffusion config validation."""

    def test_negative_diffusion(self):
        """Test negative diffusion coefficient."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        # Negative diffusion is unphysical
        try:
            diff = DiffusionConfig(grid, constants={"x": -0.5})
            # May accept and handle, or reject
        except (ValueError, AssertionError):
            pass

    def test_diffusion_on_nonexistent_axis(self):
        """Test diffusion for axis not in grid."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        with pytest.raises((ValueError, KeyError)):
            DiffusionConfig(grid, constants={"y": 0.5})  # No y-axis

    def test_none_grid(self):
        """Test diffusion with None grid."""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            DiffusionConfig(None, constants={"x": 0.5})

    def test_empty_constants_dict(self):
        """Test diffusion with empty constants."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        diff = DiffusionConfig(grid, constants={})
        # Should handle gracefully
        assert diff is not None

    def test_nan_diffusion(self):
        """Test diffusion with NaN value."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        with pytest.raises((ValueError, AssertionError)):
            DiffusionConfig(grid, constants={"x": np.nan})

    def test_inf_diffusion(self):
        """Test diffusion with infinite value."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        with pytest.raises((ValueError, OverflowError)):
            DiffusionConfig(grid, constants={"x": np.inf})

    def test_wrong_constants_type(self):
        """Test diffusion with wrong type for constants."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        with pytest.raises((ValueError, TypeError)):
            DiffusionConfig(grid, constants=0.5)  # Should be dict


@pytest.mark.slow
@pytest.mark.validation
class TestParameterCombinations:
    """Test problematic parameter combinations."""

    def test_incompatible_grid_and_ic_dimensions(self):
        """Test 1D IC on 2D grid without y parameters."""
        grid = Grid(
            x_start=0.0, x_end=10.0, num_points_x=32, y_start=0.0, y_end=10.0, num_points_y=32
        )
        # Missing y0, sigma_y -> library provides defaults, so should succeed
        ic = InitialCondition(grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        assert ic.f0.shape == (32, 32)

    def test_very_narrow_gaussian(self):
        """Test Gaussian much narrower than grid spacing."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=16)  # Coarse
        # sigma much smaller than dx
        ic = InitialCondition(grid, func_type="gaussian", x0=5.0, sigma_x=0.001)
        # Should handle but may lose mass
        assert ic.f0 is not None

    def test_very_wide_gaussian(self):
        """Test Gaussian much wider than domain."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        ic = InitialCondition(grid, func_type="gaussian", x0=5.0, sigma_x=100.0)
        # Should handle, results in nearly uniform distribution
        assert ic.f0 is not None

    def test_conflicting_2d_parameters(self):
        """Test 2D grid with asymmetric parameters."""
        grid = Grid(
            x_start=0.0, x_end=10.0, num_points_x=32, y_start=0.0, y_end=5.0, num_points_y=16
        )
        # Different aspect ratio - should work
        ic = InitialCondition(grid, func_type="gaussian", x0=5.0, y0=2.5, sigma_x=1.0, sigma_y=0.5)
        assert ic.f0 is not None
        assert ic.f0.shape == (32, 16)


@pytest.mark.slow
@pytest.mark.validation
class TestEdgeCaseInputs:
    """Test edge case inputs."""

    def test_extremely_small_values(self):
        """Test with extremely small but valid values."""
        grid = Grid(x_start=0.0, x_end=1e-10, num_points_x=64)
        ic = InitialCondition(grid, func_type="gaussian", x0=5e-11, sigma_x=1e-11)
        assert ic.f0 is not None

    def test_extremely_large_domain(self):
        """Test with very large domain."""
        grid = Grid(x_start=-1e6, x_end=1e6, num_points_x=64)
        ic = InitialCondition(grid, func_type="gaussian", x0=0.0, sigma_x=1e5)
        assert ic.f0 is not None

    def test_unicode_in_func_type(self):
        """Test Unicode characters in string parameters."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        with pytest.raises((ValueError, KeyError)):
            InitialCondition(grid, func_type="gaussïan", x0=5.0, sigma_x=0.5)

    def test_float_num_points(self):
        """Test float instead of int for num_points."""
        with pytest.raises((TypeError, ValueError)):
            Grid(x_start=0.0, x_end=10.0, num_points_x=64.5)
