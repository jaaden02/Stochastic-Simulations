"""Run tests without pytest - much faster!"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from stochlib.setup import Grid, InitialCondition, VelocitiesConfig, DiffusionConfig
from stochlib.boundary_conditions import BoundaryConditions

def test_grid_creation():
    """Test basic 1D grid creation."""
    grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
    assert grid.x_start == 0.0
    assert grid.x_end == 10.0
    assert grid.num_points_x == 64
    print("✓ test_grid_creation passed")

def test_2d_grid_creation():
    """Test 2D grid creation."""
    grid = Grid(
        x_start=0.0, x_end=10.0, num_points_x=32,
        y_start=-5.0, y_end=5.0, num_points_y=32,
    )
    assert grid.x_start == 0.0
    assert grid.x_end == 10.0
    assert grid.num_points_x == 32
    assert grid.y_start == -5.0
    assert grid.y_end == 5.0
    assert grid.num_points_y == 32
    print("✓ test_2d_grid_creation passed")

def test_grid_spacing():
    """Test that 1D grid spacing is uniform."""
    grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
    expected_dx = 10.0 / (64 - 1)
    assert hasattr(grid, 'dx')
    assert np.isclose(grid.dx, expected_dx)
    print("✓ test_grid_spacing passed")

def test_gaussian_ic():
    """Test Gaussian initial condition."""
    grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
    ic = InitialCondition(grid, func_type='gaussian', x0=5.0, sigma_x=0.5)
    assert ic.f0 is not None
    assert len(ic.f0) == grid.num_points_x
    print("✓ test_gaussian_ic passed")

def test_ic_normalized():
    """Test that Gaussian IC integrates to 1."""
    grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
    ic = InitialCondition(grid, func_type='gaussian', x0=5.0, sigma_x=0.5)
    integral = np.trapz(ic.f0, dx=grid.dx)
    assert np.isclose(integral, 1.0, rtol=0.01)
    print("✓ test_ic_normalized passed")

def test_boundary_conditions():
    """Test boundary condition creation."""
    grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
    bc = BoundaryConditions(grid, bc_x=BoundaryConditions.PERIODIC)
    assert bc is not None
    print("✓ test_boundary_conditions passed")

def test_velocities_config():
    """Test velocities configuration."""
    grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
    vel = VelocitiesConfig(grid, mu_x=0.5)
    assert vel is not None
    print("✓ test_velocities_config passed")

def test_diffusion_config():
    """Test diffusion configuration."""
    grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
    diff = DiffusionConfig(grid, constants={'x': 0.1})
    assert diff is not None
    print("✓ test_diffusion_config passed")

if __name__ == "__main__":
    print("Running fast standalone tests...\n")
    
    tests = [
        test_grid_creation,
        test_2d_grid_creation,
        test_grid_spacing,
        test_gaussian_ic,
        test_ic_normalized,
        test_boundary_conditions,
        test_velocities_config,
        test_diffusion_config,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    
    sys.exit(0 if failed == 0 else 1)
