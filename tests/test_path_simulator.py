"""Tests for SDE path simulation components."""
import pytest
import numpy as np
from stochlib import PathSimulator, StepSchemeAdvisor, PathDiagnostics, SimulationSetup
from stochlib import Grid, VelocitiesConfig, DiffusionConfig, BoundaryConditions


class TestPathSimulator:
    """Test PathSimulator for various SDE scenarios."""

    def test_constant_drift_constant_diffusion_1d(self):
        """Test 1D Ornstein-Uhlenbeck with constant coefficients."""
        # dX = -0.5*X*dt + 0.3*dW
        def drift(x, t):
            return -0.5 * x
        def diffusion(x, t):
            return np.full_like(x, 0.3)
        
        sim = PathSimulator(drift=drift, diffusion=diffusion, scheme="euler_maruyama", rng=np.random.default_rng(42))
        x0 = np.array([1.0])
        t_array = np.linspace(0, 2.0, 100)
        result = sim.simulate(x0=x0, t_array=t_array, n_paths=50)
        
        assert "paths" in result
        assert "mean" in result
        assert "var" in result
        assert result["paths"].shape == (50, 100, 1)
        assert result["mean"].shape == (100, 1)
        assert result["var"].shape == (100, 1)
        # Mean should decay exponentially (roughly)
        assert result["mean"][-1, 0] < result["mean"][0, 0]

    def test_2d_paths(self):
        """Test 2D system with independent noise."""
        def drift(x, t):
            return -0.1 * x
        def diffusion(x, t):
            return np.full_like(x, 0.2)
        
        sim = PathSimulator(drift=drift, diffusion=diffusion, rng=np.random.default_rng(123))
        x0 = np.array([1.0, -0.5])
        t_array = np.linspace(0, 1.0, 50)
        result = sim.simulate(x0=x0, t_array=t_array, n_paths=20)
        
        assert result["paths"].shape == (20, 50, 2)
        assert result["mean"].shape == (50, 2)
        assert result["var"].shape == (50, 2)

    def test_multiple_initial_conditions(self):
        """Test with multiple initial conditions (n_paths inferred)."""
        def drift(x, t):
            return np.zeros_like(x)
        def diffusion(x, t):
            return np.ones_like(x)
        
        sim = PathSimulator(drift=drift, diffusion=diffusion, rng=np.random.default_rng(99))
        x0 = np.array([[0.0], [1.0], [2.0]])  # 3 paths, dim=1
        t_array = np.linspace(0, 0.5, 20)
        result = sim.simulate(x0=x0, t_array=t_array)
        
        assert result["paths"].shape == (3, 20, 1)

    def test_save_paths_false(self):
        """Test that save_paths=False omits full path storage."""
        def drift(x, t):
            return -x
        def diffusion(x, t):
            return 0.1 * np.ones_like(x)
        
        sim = PathSimulator(drift=drift, diffusion=diffusion, rng=np.random.default_rng(77))
        x0 = np.array([1.0])
        t_array = np.linspace(0, 1.0, 100)
        result = sim.simulate(x0=x0, t_array=t_array, n_paths=100, save_paths=False)
        
        assert "paths" not in result
        assert "mean" in result
        assert "var" in result

    def test_milstein_scheme(self):
        """Test Milstein scheme with state-dependent diffusion."""
        def drift(x, t):
            return np.zeros_like(x)
        def diffusion(x, t):
            return 0.5 + 0.1 * x  # state-dependent
        def diffusion_jacobian(x, t):
            return 0.1 * np.ones_like(x)
        
        sim = PathSimulator(
            drift=drift,
            diffusion=diffusion,
            diffusion_jacobian=diffusion_jacobian,
            scheme="milstein",
            rng=np.random.default_rng(55),
        )
        x0 = np.array([1.0])
        t_array = np.linspace(0, 1.0, 50)
        result = sim.simulate(x0=x0, t_array=t_array, n_paths=30)
        
        assert result["paths"].shape == (30, 50, 1)
        assert sim.scheme == "milstein"

    def test_diagnostics_collection(self):
        """Test that diagnostics are collected during simulation."""
        def drift(x, t):
            return -0.5 * x
        def diffusion(x, t):
            return 0.2 * np.ones_like(x)
        
        diag = PathDiagnostics()
        sim = PathSimulator(drift=drift, diffusion=diffusion, rng=np.random.default_rng(88))
        x0 = np.array([1.0])
        t_array = np.linspace(0, 1.0, 20)
        result = sim.simulate(x0=x0, t_array=t_array, n_paths=10, diagnostics=diag)
        
        assert len(diag.history["mean"]) == len(t_array) - 1  # recorded at each step
        assert len(diag.history["var"]) == len(t_array) - 1

    def test_invalid_x0_shape(self):
        """Test error on invalid x0 shape."""
        def drift(x, t):
            return -x
        def diffusion(x, t):
            return 0.1 * np.ones_like(x)
        
        sim = PathSimulator(drift=drift, diffusion=diffusion)
        x0 = np.ones((2, 2, 2))  # 3D not allowed
        t_array = np.linspace(0, 1, 10)
        
        with pytest.raises(ValueError, match="x0 must be shape"):
            sim.simulate(x0=x0, t_array=t_array)

    def test_invalid_t_array(self):
        """Test error on non-monotonic t_array."""
        def drift(x, t):
            return -x
        def diffusion(x, t):
            return 0.1 * np.ones_like(x)
        
        sim = PathSimulator(drift=drift, diffusion=diffusion)
        x0 = np.array([1.0])
        t_array = np.array([0.0, 1.0, 0.5])  # not monotonic
        
        with pytest.raises(ValueError, match="strictly increasing"):
            sim.simulate(x0=x0, t_array=t_array, n_paths=10)


class TestStepSchemeAdvisor:
    """Test scheme selection logic."""

    def test_choose_auto_constant_diffusion(self):
        """Auto should pick euler_maruyama for constant diffusion."""
        def diffusion(x, t):
            return 0.5 * np.ones_like(x)
        
        scheme = StepSchemeAdvisor.choose_scheme("auto", diffusion, None)
        assert scheme == "euler_maruyama"

    def test_choose_auto_state_dependent(self):
        """Auto should pick milstein for state-dependent diffusion."""
        def diffusion(x, t):
            return 0.5 + 0.2 * x
        
        scheme = StepSchemeAdvisor.choose_scheme("auto", diffusion, None)
        assert scheme == "milstein"

    def test_choose_auto_with_jacobian(self):
        """Auto should pick milstein if jacobian provided."""
        def diffusion(x, t):
            return 0.5 * np.ones_like(x)
        def diffusion_jac(x, t):
            return np.zeros_like(x)
        
        scheme = StepSchemeAdvisor.choose_scheme("auto", diffusion, diffusion_jac)
        assert scheme == "milstein"

    def test_choose_manual_override(self):
        """Manual scheme selection should be respected."""
        def diffusion(x, t):
            return 0.5 + 0.2 * x
        
        scheme = StepSchemeAdvisor.choose_scheme("euler_maruyama", diffusion, None)
        assert scheme == "euler_maruyama"

    def test_invalid_scheme(self):
        """Test error on unknown scheme."""
        def diffusion(x, t):
            return 0.5 * np.ones_like(x)
        
        with pytest.raises(ValueError, match="Unknown scheme"):
            StepSchemeAdvisor.choose_scheme("bogus", diffusion, None)


class TestBoundaryHandling:
    """Test boundary handling modes."""
    
    def test_absorb_mode(self):
        """Test absorbing boundaries: paths stop at boundary."""
        def drift(x, t):
            return np.full_like(x, 1.0)  # drift to the right
        def diffusion(x, t):
            return 0.01 * np.ones_like(x)
        
        bounds = (np.array([0.0]), np.array([2.0]))
        sim = PathSimulator(
            drift=drift,
            diffusion=diffusion,
            bounds=bounds,
            boundary_mode="absorb",
            rng=np.random.default_rng(42),
        )
        
        x0 = np.array([1.0])
        t_array = np.linspace(0, 5.0, 100)
        result = sim.simulate(x0=x0, t_array=t_array, n_paths=50)
        
        # Some paths should exit
        assert np.any(~np.isnan(result["exit_times"]))
        # Number alive should decrease over time
        assert result["n_alive"][-1] < result["n_alive"][0]
        # Alive mask should track exits
        assert result["alive_mask"].shape == (50, 100)
    
    def test_reject_mode(self):
        """Test reject mode: paths that exit are entirely invalid."""
        def drift(x, t):
            return np.full_like(x, 0.5)
        def diffusion(x, t):
            return 0.1 * np.ones_like(x)
        
        bounds = (np.array([0.0]), np.array([1.5]))
        sim = PathSimulator(
            drift=drift,
            diffusion=diffusion,
            bounds=bounds,
            boundary_mode="reject",
            rng=np.random.default_rng(77),
        )
        
        x0 = np.array([0.5])
        t_array = np.linspace(0, 3.0, 50)
        result = sim.simulate(x0=x0, t_array=t_array, n_paths=30)
        
        # Once a path exits, it should be dead for all remaining time
        for path_idx in range(30):
            alive_history = result["alive_mask"][path_idx, :]
            if not np.all(alive_history):
                # Once dead, stays dead
                first_dead = np.where(~alive_history)[0][0]
                assert np.all(~alive_history[first_dead:])
    
    def test_no_boundaries(self):
        """Test that no boundary handling works (infinite domain)."""
        def drift(x, t):
            return -0.5 * x
        def diffusion(x, t):
            return 0.3 * np.ones_like(x)
        
        sim = PathSimulator(drift=drift, diffusion=diffusion, rng=np.random.default_rng(99))
        x0 = np.array([10.0])  # start far from origin
        t_array = np.linspace(0, 2.0, 50)
        result = sim.simulate(x0=x0, t_array=t_array, n_paths=20)
        
        # All paths should remain alive
        assert np.all(result["alive_mask"])
        assert np.all(result["n_alive"] == 20)
        assert np.all(np.isnan(result["exit_times"]))
    
    def test_2d_boundaries(self):
        """Test 2D boundaries (rectangular domain)."""
        def drift(x, t):
            return np.zeros_like(x)
        def diffusion(x, t):
            return 0.5 * np.ones_like(x)
        
        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        sim = PathSimulator(
            drift=drift,
            diffusion=diffusion,
            bounds=bounds,
            boundary_mode="absorb",
            rng=np.random.default_rng(55),
        )
        
        x0 = np.array([0.5, 0.5])  # center
        t_array = np.linspace(0, 2.0, 100)
        result = sim.simulate(x0=x0, t_array=t_array, n_paths=40)
        
        # Some paths should exit the square
        assert np.any(~np.isnan(result["exit_times"]))
    
    def test_boundary_mode_requires_bounds(self):
        """Test that boundary_mode requires bounds to be specified."""
        def drift(x, t):
            return -x
        def diffusion(x, t):
            return 0.1 * np.ones_like(x)
        
        with pytest.raises(ValueError, match="bounds required"):
            PathSimulator(drift=drift, diffusion=diffusion, boundary_mode="absorb")
    
    def test_invalid_boundary_mode(self):
        """Test error on invalid boundary mode."""
        def drift(x, t):
            return -x
        def diffusion(x, t):
            return 0.1 * np.ones_like(x)
        
        bounds = (np.array([0.0]), np.array([1.0]))
        with pytest.raises(ValueError, match="boundary_mode must be"):
            PathSimulator(drift=drift, diffusion=diffusion, bounds=bounds, boundary_mode="invalid")


class TestPathDiagnostics:
    """Test path diagnostics tracking."""

    def test_record_statistics(self):
        """Test that statistics are recorded correctly."""
        diag = PathDiagnostics()
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 paths, 2D
        diag.record(x, t=0.5)
        
        assert len(diag.history["mean"]) == 1
        assert len(diag.history["var"]) == 1
        assert len(diag.history["min"]) == 1
        assert len(diag.history["max"]) == 1
        
        np.testing.assert_array_almost_equal(diag.history["mean"][0], [3.0, 4.0])


class TestSimulationSetup:
    """Test the SimulationSetup factory."""

    def test_fokker_planck_mode(self):
        """Test that FP mode returns SimulationEngine."""
        grid = Grid(x_start=0, x_end=1, num_points_x=10)
        velocities = VelocitiesConfig(grid, mu_x=0.1)
        diffusions = DiffusionConfig(grid, constants={'x': 0.01})
        bc = BoundaryConditions(grid, bc_x='periodic')
        
        setup = SimulationSetup(
            mode="fokker_planck",
            grid=grid,
            velocities=velocities,
            diffusions=diffusions,
            boundary_conditions=bc,
        )
        engine = setup.build()
        
        from stochlib.fokker_planck.selector import SimulationEngine
        assert isinstance(engine, SimulationEngine)

    def test_paths_mode(self):
        """Test that paths mode returns PathSimulator."""
        def drift(x, t):
            return -0.5 * x
        def diffusion(x, t):
            return 0.3 * np.ones_like(x)
        
        setup = SimulationSetup(
            mode="paths",
            drift=drift,
            diffusion=diffusion,
        )
        sim = setup.build()
        
        assert isinstance(sim, PathSimulator)

    def test_fokker_planck_missing_args(self):
        """Test error when FP mode missing required args."""
        setup = SimulationSetup(mode="fokker_planck")
        
        with pytest.raises(ValueError, match="required for Fokker-Planck"):
            setup.build()

    def test_paths_missing_args(self):
        """Test error when paths mode missing required args."""
        setup = SimulationSetup(mode="paths")
        
        with pytest.raises(ValueError, match="required for path simulation"):
            setup.build()

    def test_invalid_mode(self):
        """Test error on unknown mode."""
        setup = SimulationSetup(mode="bogus")
        
        with pytest.raises(ValueError, match="Unknown simulation mode"):
            setup.build()
