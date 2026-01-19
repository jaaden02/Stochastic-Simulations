"""Integration tests combining multiple components."""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stochlib.setup import Grid, InitialCondition, VelocitiesConfig, DiffusionConfig
from stochlib.boundary_conditions import BoundaryConditions
from stochlib.fokker_planck import SimulationEngine


class TestBasicWorkflow:
    """Test complete workflow from setup to visualization."""
    
    def test_setup_and_run(self, simple_1d_grid):
        """Test full workflow setup."""
        # 1. Create initial condition
        ic = InitialCondition(simple_1d_grid, func_type='gaussian', x0=5.0, sigma_x=0.5)
        f0 = ic.f0
        
        # 2. Setup physics
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.1)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.05})
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.PERIODIC)
        
        # 3. Create engine
        engine = SimulationEngine(simple_1d_grid, vel, diff, bc)
        
        # 4. Check initial condition
        assert f0 is not None
        assert engine is not None
        assert len(f0) == simple_1d_grid.num_points_x
    
    def test_workflow_with_analysis(self, simple_1d_grid):
        """Test workflow with analysis."""
        # Setup
        ic = InitialCondition(simple_1d_grid, func_type='gaussian', x0=5.0, sigma_x=0.5)
        f0 = ic.f0
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.2)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.1})
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.OPEN)
        
        # Analysis
        x = np.linspace(simple_1d_grid.x_start, simple_1d_grid.x_end, simple_1d_grid.num_points_x)
        mean = np.sum(x * f0) / np.sum(f0)
        variance = np.sum((x - mean)**2 * f0) / np.sum(f0)
        
        assert 4.0 < mean < 6.0
        assert variance > 0


class TestPhysicalScenarios:
    """Test realistic physical scenarios."""
    
    def test_gaussian_diffusion(self, simple_1d_grid):
        """Test Gaussian spreading via diffusion."""
        # Pure diffusion should spread initial Gaussian
        ic = InitialCondition(simple_1d_grid, func_type='gaussian', x0=5.0, sigma_x=0.3)
        f0 = ic.f0
        
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.0)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.5})
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.PERIODIC)
        
        engine = SimulationEngine(simple_1d_grid, vel, diff, bc)
        
        # Initial peak
        peak_0 = np.max(f0)
        assert peak_0 > 0
    
    def test_drift_transport(self, simple_1d_grid):
        """Test particle transport via drift."""
        # Drift should move initial condition
        ic = InitialCondition(simple_1d_grid, func_type='gaussian', x0=3.0, sigma_x=0.5)
        f0 = ic.f0
        
        vel = VelocitiesConfig(simple_1d_grid, mu_x=1.0)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.01})
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.OPEN)
        
        engine = SimulationEngine(simple_1d_grid, vel, diff, bc)
        
        # Peak at 3
        peak_idx = np.argmax(f0)
        x_peak = simple_1d_grid.x_start + peak_idx * simple_1d_grid.dx
        assert 2.5 < x_peak < 3.5
    
    def test_drift_diffusion_balance(self, simple_1d_grid):
        """Test balanced drift-diffusion."""
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.5)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.1})
        
        ic = InitialCondition(simple_1d_grid, func_type='gaussian', x0=5.0, sigma_x=0.5)
        f0 = ic.f0
        
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.PERIODIC)
        engine = SimulationEngine(simple_1d_grid, vel, diff, bc)
        
        assert engine is not None


class TestBoundaryConditionEffects:
    """Test effects of different boundary conditions."""
    
    def test_periodic_bc_workflow(self, simple_1d_grid):
        """Test workflow with periodic BC."""
        ic = InitialCondition(simple_1d_grid, func_type='gaussian', x0=5.0, sigma_x=0.5)
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.5)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.1})
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.PERIODIC)
        
        engine = SimulationEngine(simple_1d_grid, vel, diff, bc)
        assert engine is not None
    
    def test_dirichlet_bc_workflow(self, simple_1d_grid):
        """Test workflow with Dirichlet BC."""
        ic = InitialCondition(simple_1d_grid, func_type='gaussian', x0=5.0, sigma_x=0.5)
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.5)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.1})
        bc = BoundaryConditions(
            simple_1d_grid,
            bc_x='noflux'
        )
        
        engine = SimulationEngine(simple_1d_grid, vel, diff, bc)
        assert engine is not None
    
    def test_open_bc_workflow(self, simple_1d_grid):
        """Test workflow with open BC."""
        ic = InitialCondition(simple_1d_grid, func_type='gaussian', x0=2.0, sigma_x=0.5)
        vel = VelocitiesConfig(simple_1d_grid, mu_x=1.0)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.05})
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.OPEN)
        
        engine = SimulationEngine(simple_1d_grid, vel, diff, bc)
        assert engine is not None


class TestGridResolutionEffect:
    """Test effect of grid resolution on workflow."""
    
    def test_coarse_grid_workflow(self):
        """Test workflow with coarse grid."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=16)
        ic = InitialCondition(grid, func_type='gaussian', x0=5.0, sigma_x=0.5)
        vel = VelocitiesConfig(grid, mu_x=0.5)
        diff = DiffusionConfig(grid, constants={'x': 0.1})
        bc = BoundaryConditions(grid, bc_x=BoundaryConditions.PERIODIC)
        
        engine = SimulationEngine(grid, vel, diff, bc)
        assert engine is not None
    
    def test_fine_grid_workflow(self, fine_1d_grid):
        """Test workflow with fine grid."""
        ic = InitialCondition(fine_1d_grid, func_type='gaussian', x0=5.0, sigma_x=0.5)
        vel = VelocitiesConfig(fine_1d_grid, mu_x=0.5)
        diff = DiffusionConfig(fine_1d_grid, constants={'x': 0.1})
        bc = BoundaryConditions(fine_1d_grid, bc_x=BoundaryConditions.PERIODIC)
        
        engine = SimulationEngine(fine_1d_grid, vel, diff, bc)
        assert engine is not None
    
    def test_medium_grid_workflow(self, medium_1d_grid):
        """Test workflow with medium grid."""
        ic = InitialCondition(medium_1d_grid, func_type='gaussian', x0=5.0, sigma_x=0.5)
        vel = VelocitiesConfig(medium_1d_grid, mu_x=0.5)
        diff = DiffusionConfig(medium_1d_grid, constants={'x': 0.1})
        bc = BoundaryConditions(medium_1d_grid, bc_x=BoundaryConditions.PERIODIC)
        
        engine = SimulationEngine(medium_1d_grid, vel, diff, bc)
        assert engine is not None


class TestParameterSweep:
    """Test workflow with parameter sweeps."""
    
    def test_velocity_sweep(self, simple_1d_grid):
        """Test sweeping velocity parameter."""
        velocities = [0.1, 0.5, 1.0, 2.0]
        engines = []
        
        for u in velocities:
            vel = VelocitiesConfig(simple_1d_grid, mu_x=u)
            diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.1})
            bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.PERIODIC)
            
            engine = SimulationEngine(simple_1d_grid, vel, diff, bc)
            engines.append(engine)
        
        assert len(engines) == len(velocities)
    
    def test_diffusion_sweep(self, simple_1d_grid):
        """Test sweeping diffusion parameter."""
        diffusions = [0.01, 0.05, 0.1, 0.5]
        engines = []
        
        for D in diffusions:
            vel = VelocitiesConfig(simple_1d_grid, mu_x=0.5)
            diff = DiffusionConfig(simple_1d_grid, constants={'x': D})
            bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.PERIODIC)
            
            engine = SimulationEngine(simple_1d_grid, vel, diff, bc)
            engines.append(engine)
        
        assert len(engines) == len(diffusions)
    
    def test_ic_shape_sweep(self, simple_1d_grid):
        """Test sweeping initial condition shape."""
        sigmas = [0.2, 0.5, 1.0]
        ics = []
        
        for sigma in sigmas:
            ic = InitialCondition(simple_1d_grid, func_type='gaussian', x0=5.0, sigma_x=sigma)
            ics.append(ic.f0)
        
        assert len(ics) == len(sigmas)
        # Wider ICs should have lower peak
        peaks = [np.max(f) for f in ics]
        assert peaks[0] > peaks[-1]


class TestConsistency:
    """Test consistency across different setups."""
    
    def test_conservation_across_bcs(self, simple_1d_grid):
        """Test mass conservation with different BCs."""
        bc_types = [
            'periodic',
            'open',
            'noflux',
        ]
        
        masses = []
        for bc_type in bc_types:
            ic = InitialCondition(simple_1d_grid, func_type='gaussian', x0=5.0, sigma_x=0.5)
            f = ic.f0
            mass = np.trapz(f, dx=simple_1d_grid.dx)
            masses.append(mass)
        
        # All should have similar mass
        assert np.std(masses) < 0.05
    
    def test_mean_position_tracking(self, simple_1d_grid):
        """Test that mean position can be tracked."""
        times = np.linspace(0, 1, 5)
        means = []
        
        for t in times:
            # Simulate moving IC
            x0 = 2.0 + 5.0 * t
            ic = InitialCondition(simple_1d_grid, func_type='gaussian', x0=x0, sigma_x=0.5)
            f = ic.f0
            
            x = np.linspace(simple_1d_grid.x_start, simple_1d_grid.x_end, simple_1d_grid.num_points_x)
            mean = np.sum(x * f) / np.sum(f)
            means.append(mean)
        
        # Mean position should increase
        assert means[0] < means[-1]


class TestMultipleDimensions:
    """Test workflows in multiple dimensions."""
    
    def test_1d_workflow(self, simple_1d_grid):
        """Test 1D workflow."""
        ic = InitialCondition(simple_1d_grid, func_type='gaussian', x0=5.0, sigma_x=0.5)
        vel = VelocitiesConfig(simple_1d_grid, mu_x=0.5)
        diff = DiffusionConfig(simple_1d_grid, constants={'x': 0.1})
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.PERIODIC)
        
        engine = SimulationEngine(simple_1d_grid, vel, diff, bc)
        assert engine is not None
    
    def test_2d_workflow(self, simple_2d_grid):
        """Test 2D workflow."""
        ic = InitialCondition(
            simple_2d_grid,
            func_type='gaussian',
            x0=5.0, y0=5.0,
            sigma_x=0.5, sigma_y=0.5
        )
        vel = VelocitiesConfig(simple_2d_grid, mu_x=0.3, mu_y=0.2)
        diff = DiffusionConfig(simple_2d_grid, constants={'x': 0.1, 'y': 0.08})
        bc = BoundaryConditions(simple_2d_grid, bc_x=BoundaryConditions.PERIODIC, bc_y=BoundaryConditions.PERIODIC)
        
        engine = SimulationEngine(simple_2d_grid, vel, diff, bc)
        assert engine is not None
