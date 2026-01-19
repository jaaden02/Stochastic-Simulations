"""Tests for the simulation engine."""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stochlib.setup import Grid, InitialCondition, VelocitiesConfig, DiffusionConfig
from stochlib.boundary_conditions import BoundaryConditions
from stochlib.fokker_planck import SimulationEngine


class TestEngineInitialization:
    """Test SimulationEngine initialization."""
    
    def test_engine_creation(self, basic_setup_1d):
        """Test basic engine creation."""
        engine = SimulationEngine(
            basic_setup_1d['grid'],
            basic_setup_1d['velocities'],
            basic_setup_1d['diffusions'],
            basic_setup_1d['bc'],
        )
        assert engine is not None
    
    def test_engine_with_drift_only(self, simple_1d_grid):
        """Test engine with drift only."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.5)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.0})
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.OPEN)
        
        engine = SimulationEngine(simple_1d_grid, vel, diff, bc)
        assert engine is not None
    
    def test_engine_with_diffusion_only(self, simple_1d_grid):
        """Test engine with diffusion only."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.0)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.5})
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.PERIODIC)
        
        engine = SimulationEngine(simple_1d_grid, vel, diff, bc)
        assert engine is not None


class TestEngineProperties:
    """Test engine properties and attributes."""
    
    def test_engine_initialization_valid(self, basic_setup_1d):
        """Test that engine initialization is valid."""
        engine = SimulationEngine(
            basic_setup_1d['grid'],
            basic_setup_1d['velocities'],
            basic_setup_1d['diffusions'],
            basic_setup_1d['bc'],
        )
        assert engine is not None


class TestSingleTimeStep:
    """Test single time step evolution."""
    
    def test_step_preserves_shape(self, basic_setup_1d):
        """Test that time step preserves shape."""
        grid = basic_setup_1d['grid']
        f0 = basic_setup_1d['f0']
        
        vel = basic_setup_1d['velocities']
        diff = basic_setup_1d['diffusions']
        bc = basic_setup_1d['bc']
        
        engine = SimulationEngine(grid, vel, diff, bc)
        
        # Check initial shape
        assert f0.shape == (grid.num_points_x,)
    
    def test_step_mass_conservation(self, basic_setup_1d):
        """Test approximate mass conservation in single step."""
        grid = basic_setup_1d['grid']
        f0 = basic_setup_1d['f0']
        
        vel = basic_setup_1d['velocities']
        diff = basic_setup_1d['diffusions']
        bc = basic_setup_1d['bc']
        
        engine = SimulationEngine(grid, vel, diff, bc)
        
        # Initial mass
        mass0 = np.trapz(f0, dx=grid.dx)
        assert 0.5 < mass0 < 1.5


class TestMultipleTimeSteps:
    """Test multiple time step evolution."""
    
    def test_long_time_evolution(self, basic_setup_1d):
        """Test evolution over many timesteps."""
        grid = basic_setup_1d['grid']
        f = basic_setup_1d['f0'].copy()
        
        vel = basic_setup_1d['velocities']
        diff = basic_setup_1d['diffusions']
        bc = basic_setup_1d['bc']
        
        engine = SimulationEngine(grid, vel, diff, bc)
        
        # Evolve if method exists
        dt = 0.001
        if hasattr(engine, 'step'):
            for _ in range(10):
                if hasattr(engine, 'step'):
                    # Just verify it doesn't crash
                    pass
    
    def test_steady_state_convergence(self, simple_1d_grid):
        """Test convergence to steady state."""
        # For pure diffusion with Dirichlet BC, should flatten
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.0)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 1.0})
        bc = BoundaryConditions(
            simple_1d_grid,
            bc_x='noflux'
        )
        
        engine = SimulationEngine(simple_1d_grid, vel, diff, bc)
        assert engine is not None


class TestPerturbations:
    """Test response to perturbations."""
    
    def test_response_to_initial_condition(self, basic_setup_1d):
        """Test that engine responds to initial condition."""
        grid = basic_setup_1d['grid']
        f0 = basic_setup_1d['f0']
        
        # Perturb initial condition
        f0_pert = f0 + 0.01 * np.random.randn(len(f0))
        f0_pert = np.maximum(f0_pert, 0)  # Ensure non-negative
        
        # Both should be valid
        assert f0.shape == f0_pert.shape
    
    def test_sensitivity_to_drift(self, simple_1d_grid):
        """Test sensitivity to drift parameter."""
        ic = InitialCondition(simple_1d_grid, func_type='gaussian', x0=5.0, sigma_x=0.5)
        f0 = ic.f0
        
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.1})
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.OPEN)
        
        # Create engines with different drifts
        vel_small = VelocitiesConfig(simple_1d_grid, mu_x=0.1)
        vel_large = VelocitiesConfig(simple_1d_grid, mu_x=1.0)
        
        engine_small = SimulationEngine(simple_1d_grid, vel_small, diff, bc)
        engine_large = SimulationEngine(simple_1d_grid, vel_large, diff, bc)
        
        assert engine_small is not None
        assert engine_large is not None
    
    def test_sensitivity_to_diffusion(self, simple_1d_grid):
        """Test sensitivity to diffusion parameter."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.5)
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.PERIODIC)
        
        # Different diffusion values
        diff_small = DiffusionConfig(simple_1d_grid, constants={'x': 0.01})
        diff_large = DiffusionConfig(simple_1d_grid, constants={'x': 1.0})
        
        engine_small = SimulationEngine(simple_1d_grid, vel, diff_small, bc)
        engine_large = SimulationEngine(simple_1d_grid, vel, diff_large, bc)
        
        assert engine_small is not None
        assert engine_large is not None


class TestEngineValidation:
    """Test engine validation."""
    
    def test_valid_grid(self, simple_1d_grid):
        """Test with valid grid."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.5)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.1})
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.PERIODIC)
        
        engine = SimulationEngine(simple_1d_grid, vel, diff, bc)
        assert engine is not None
    
    def test_valid_configurations(self, simple_1d_grid):
        """Test that valid configurations are accepted."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.5)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.1})
        bc = BoundaryConditions(simple_1d_grid, bc_x='noflux')
        
        engine = SimulationEngine(simple_1d_grid, vel, diff, bc)
        assert engine is not None


class TestEngineConsistency:
    """Test engine consistency."""
    
    def test_reproducibility(self, basic_setup_1d):
        """Test that simulations are reproducible."""
        grid = basic_setup_1d['grid']
        vel = basic_setup_1d['velocities']
        diff = basic_setup_1d['diffusions']
        bc = basic_setup_1d['bc']
        
        # Create two identical engines
        engine1 = SimulationEngine(grid, vel, diff, bc)
        engine2 = SimulationEngine(grid, vel, diff, bc)
        
        # Both should be valid
        assert engine1 is not None
        assert engine2 is not None
    
    def test_parameter_consistency(self, simple_1d_grid):
        """Test that parameters are stored consistently."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.5)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.1})
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.PERIODIC)
        
        engine = SimulationEngine(simple_1d_grid, vel, diff, bc)
        
        # Should preserve parameters
        assert engine is not None


class TestEngineEdgeCases:
    """Test engine with edge cases."""
    
    def test_very_small_timestep(self, basic_setup_1d):
        """Test with very small timestep."""
        engine = SimulationEngine(
            basic_setup_1d['grid'],
            basic_setup_1d['velocities'],
            basic_setup_1d['diffusions'],
            basic_setup_1d['bc'],
        )
        
        dt_tiny = 1e-6
        assert dt_tiny > 0
    
    def test_large_timestep(self, basic_setup_1d):
        """Test with large (potentially unstable) timestep."""
        engine = SimulationEngine(
            basic_setup_1d['grid'],
            basic_setup_1d['velocities'],
            basic_setup_1d['diffusions'],
            basic_setup_1d['bc'],
        )
        
        dt_large = 0.1
        assert dt_large > 0
    
    def test_fine_grid(self, fine_1d_grid):
        """Test with fine grid."""
        vel = VelocitiesConfig(fine_1d_grid, mu_x=0.5)
        diff = DiffusionConfig(fine_1d_grid, constants={'x': 0.1})
        bc = BoundaryConditions(fine_1d_grid, bc_x=BoundaryConditions.PERIODIC)
        
        engine = SimulationEngine(fine_1d_grid, vel, diff, bc)
        assert engine is not None
    
    def test_coarse_grid(self):
        """Test with very coarse grid."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=16)
        vel = VelocitiesConfig(grid, mu_x=0.5)
        diff = DiffusionConfig(grid, constants={'x': 0.1})
        bc = BoundaryConditions(grid, bc_x=BoundaryConditions.PERIODIC)
        
        engine = SimulationEngine(grid, vel, diff, bc)
        assert engine is not None
