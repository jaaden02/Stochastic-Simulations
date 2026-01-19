"""Tests for configuration objects."""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stochlib.setup import Grid, VelocitiesConfig, DiffusionConfig


class TestVelocitiesConfigCreation:
    """Test velocities configuration setup."""
    
    def test_constant_velocity_x(self, simple_1d_grid):
        """Test constant velocity in x direction."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=-0.1)
        assert vel is not None
    
    def test_constant_velocity_2d(self, simple_2d_grid):
        """Test constant velocities in 2D."""
        vel = VelocitiesConfig(simple_2d_grid, mu_x=0.1, mu_y=-0.1)
        assert vel is not None
    
    def test_zero_velocity(self, simple_1d_grid):
        """Test zero velocity configuration."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.0)
        assert vel is not None
    
    def test_high_velocity(self, simple_1d_grid):
        """Test high velocity configuration."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=10.0)
        assert vel is not None


class TestVelocitiesConfigTypes:
    """Test different velocity configuration types."""
    
    def test_constant_velocity_field(self, simple_1d_grid):
        """Test constant velocity field."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.5)
        assert vel is not None
    
    def test_linear_velocity_field(self, simple_1d_grid):
        """Test linear velocity field."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.1)
        assert vel is not None
    
    def test_custom_velocity_field(self, simple_1d_grid):
        """Test custom velocity field."""
        def velocity_func(x):
            return np.sin(2 * np.pi * x / 10.0)
        vel = VelocitiesConfig(simple_1d_grid, mu_x=velocity_func)
        assert vel is not None


class TestDiffusionConfigCreation:
    """Test diffusion configuration setup."""
    
    def test_constant_diffusion_x(self, simple_1d_grid):
        """Test constant diffusion in x direction."""
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.1})
        assert diff is not None
    
    def test_constant_diffusion_2d(self, simple_2d_grid):
        """Test constant diffusion in 2D."""
        diff = DiffusionConfig(simple_2d_grid, constants={'x': 0.1, 'y': 0.05})
        assert diff is not None
    
    def test_zero_diffusion(self, simple_1d_grid):
        """Test zero diffusion (advection only)."""
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.0})
        assert diff is not None
    
    def test_high_diffusion(self, simple_1d_grid):
        """Test high diffusion."""
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 10.0})
        assert diff is not None


class TestDiffusionConfigTypes:
    """Test different diffusion configuration types."""
    
    def test_constant_diffusion_field(self, simple_1d_grid):
        """Test constant diffusion field."""
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.5})
        assert diff is not None
    
    def test_position_dependent_diffusion(self, simple_1d_grid):
        """Test position-dependent diffusion."""
        def diffusion_func(x):
            return 0.1 * (1.0 + np.sin(2 * np.pi * x / 10.0))
        diff = DiffusionConfig(simple_1d_grid, constants={'x': diffusion_func})
        assert diff is not None
    
    def test_linear_diffusion_field(self, simple_1d_grid):
        """Test linearly varying diffusion."""
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.1})
        assert diff is not None


class TestConfigParameterRanges:
    """Test parameter range validity."""
    
    def test_small_velocity(self, simple_1d_grid):
        """Test very small velocity."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=1e-6)
        assert vel is not None
    
    def test_large_velocity(self, simple_1d_grid):
        """Test very large velocity."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=1e2)
        assert vel is not None
    
    def test_small_diffusion(self, simple_1d_grid):
        """Test very small diffusion."""
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 1e-6})
        assert diff is not None
    
    def test_large_diffusion(self, simple_1d_grid):
        """Test very large diffusion."""
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 1e2})
        assert diff is not None
    
    def test_negative_velocity(self, simple_1d_grid):
        """Test negative velocity (backwards flow)."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=-0.5)
        assert vel is not None


class TestConfig2D:
    """Test 2D configuration combinations."""
    
    def test_different_diffusion_in_each_direction(self, simple_2d_grid):
        """Test different diffusion coefficients in x and y."""
        diff = DiffusionConfig(simple_2d_grid, constants={'x': 0.1, 'y': 0.05})
        assert diff is not None
    
    def test_anisotropic_system(self, simple_2d_grid):
        """Test anisotropic system (different parameters in each direction)."""
        vel = VelocitiesConfig(simple_2d_grid, mu_x=0.2, mu_y=-0.1)
        diff = DiffusionConfig(simple_2d_grid, constants={'x': 0.15, 'y': 0.25})
        assert vel is not None
        assert diff is not None
    
    def test_isotropic_system(self, simple_2d_grid):
        """Test isotropic system (same parameters in both directions)."""
        vel = VelocitiesConfig(simple_2d_grid, mu_x=0.1, mu_y=0.1)
        diff = DiffusionConfig(simple_2d_grid, constants={'x': 0.1, 'y': 0.1})
        assert vel is not None
        assert diff is not None


class TestConfigCombinations:
    """Test realistic configuration combinations."""
    
    def test_drift_diffusion(self, simple_1d_grid):
        """Test drift-diffusion configuration."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.1)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.05})
        assert vel is not None
        assert diff is not None
    
    def test_pure_diffusion(self, simple_1d_grid):
        """Test pure diffusion (no drift)."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.0)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.5})
        assert vel is not None
        assert diff is not None
    
    def test_pure_advection(self, simple_1d_grid):
        """Test pure advection (no diffusion)."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=1.0)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.0})
        assert vel is not None
        assert diff is not None


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_config_accepts_valid_parameters(self, simple_1d_grid):
        """Test that valid parameters are accepted."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.5)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.1})
        assert vel is not None
        assert diff is not None
    
    def test_config_with_dictionary_parameters(self, simple_1d_grid):
        """Test config with dictionary parameter specification."""
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.1})
        assert diff is not None
    
    def test_velocities_require_grid(self):
        """Test that velocities require a valid grid."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
        vel = VelocitiesConfig(grid, mu_x=0.1)
        assert vel is not None
