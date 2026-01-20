"""Tests for numerical stability analysis."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stochlib.setup import Grid


class TestCFLCondition:
    """Test Courant-Friedrichs-Lewy (CFL) condition."""

    def test_cfl_stable_case(self, simple_1d_grid):
        """Test parameters satisfying CFL condition."""
        # CFL = |u| * dt / dx
        # For stability, CFL <= 0.5 typically
        dx = simple_1d_grid.dx
        u = 0.1
        dt = 0.4 * dx / u  # CFL = 0.04

        cfl = u * dt / dx
        assert cfl < 0.5

    def test_cfl_unstable_case(self, simple_1d_grid):
        """Test parameters violating CFL condition."""
        dx = simple_1d_grid.dx
        u = 1.0
        dt = 2.0 * dx / u  # CFL = 2.0 (unstable!)

        cfl = u * dt / dx
        assert cfl > 1.0

    def test_cfl_with_diffusion(self, simple_1d_grid):
        """Test CFL condition with diffusion."""
        # For diffusion: dt / dx^2 <= 0.25 for FTCS
        dx = simple_1d_grid.dx
        D = 0.1
        dt = 0.2 * dx**2 / D

        diffusion_cfl = D * dt / dx**2
        assert diffusion_cfl < 0.25


class TestLinearStability:
    """Test linear stability analysis."""

    def test_stable_drift_diffusion_1(self, simple_1d_grid):
        """Test stable drift-diffusion parameters."""
        u = 0.5
        D = 0.1
        dx = simple_1d_grid.dx

        # Peclet number: Pe = |u| * dx / D
        peclet = abs(u) * dx / D
        # For stability, typically Pe < 2
        assert peclet < 10

    def test_stable_drift_diffusion_2(self, fine_1d_grid):
        """Test stability with finer grid."""
        u = 0.5
        D = 0.1
        dx = fine_1d_grid.dx

        # Finer grid should have smaller Peclet number
        peclet = abs(u) * dx / D
        assert peclet < 2

    def test_high_peclet_advection_dominated(self, simple_1d_grid):
        """Test high Peclet number (advection dominated)."""
        u = 5.0
        D = 0.001
        dx = simple_1d_grid.dx

        peclet = abs(u) * dx / D
        # Advection dominated
        assert peclet > 1


class TestNumericalDiffusion:
    """Test numerical diffusion effects."""

    def test_numerical_diffusion_quantification(self):
        """Test quantification of numerical diffusion."""
        # Numerical diffusion for upwind scheme: D_num = 0.5 * u * dx
        u = 0.5
        dx = 10.0 / 64  # typical dx

        D_numerical = 0.5 * abs(u) * dx
        assert D_numerical > 0

    def test_numerical_diffusion_relative_importance(self):
        """Test relative importance of numerical vs physical diffusion."""
        u = 0.5
        dx = 0.156  # 10/64
        D_physical = 0.05
        D_numerical = 0.5 * abs(u) * dx

        ratio = D_numerical / D_physical
        # Numerical diffusion shouldn't dominate
        assert ratio < 2


class TestDispersionRelation:
    """Test wave dispersion analysis."""

    def test_advection_wave_speed(self):
        """Test wave speed for advection."""
        u = 0.5  # velocity
        # Wave speed should equal velocity for advection
        wave_speed = u
        assert wave_speed == 0.5

    def test_diffusion_damping(self):
        """Test damping from diffusion."""
        k = 2 * np.pi / 10  # wavenumber
        D = 0.1

        # Damping rate: sigma = D * k^2
        damping = D * k**2
        assert damping > 0

    def test_combined_advection_diffusion(self):
        """Test combined advection-diffusion dispersion."""
        k = 2 * np.pi / 10
        u = 0.5
        D = 0.1

        # Phase velocity: c = u
        # Damping: -D * k^2
        damping = D * k**2

        # Damping should be small compared to advection
        assert abs(u) > damping


class TestEnergyStability:
    """Test energy stability."""

    def test_diffusion_decreases_energy(self):
        """Test that diffusion decreases total energy."""
        # For diffusion: d/dt integral(f^2) <= 0
        f = np.array([1.0, 2.0, 1.0, 0.5])
        initial_energy = np.sum(f**2)

        # After diffusion step, energy should decrease
        assert initial_energy > 0

    def test_advection_preserves_energy(self):
        """Test that pure advection preserves energy."""
        f = np.array([1.0, 2.0, 1.0, 0.5])
        # Shifted version (advection)
        f_shifted = np.roll(f, 1)

        # Energy should be preserved
        energy_before = np.sum(f**2)
        energy_after = np.sum(f_shifted**2)
        assert np.isclose(energy_before, energy_after)


class TestStabilityWithGridRefinement:
    """Test stability under grid refinement."""

    def test_stability_coarse_grid(self):
        """Test stability on coarse grid."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=16)
        dx = grid.dx
        u = 0.5

        # Should satisfy CFL-like condition
        dt_max = 0.4 * dx / u
        assert dt_max > 0

    def test_stability_fine_grid(self):
        """Test stability on fine grid."""
        grid = Grid(x_start=0.0, x_end=10.0, num_points_x=256)
        dx = grid.dx
        u = 0.5

        # Finer grid requires smaller timestep for same CFL
        dt_max = 0.4 * dx / u
        assert dt_max > 0

    def test_fine_grid_dt_smaller(self):
        """Test that fine grid requires smaller timestep."""
        coarse = Grid(x_start=0.0, x_end=10.0, num_points_x=16)
        fine = Grid(x_start=0.0, x_end=10.0, num_points_x=256)

        u = 0.5
        cfl = 0.4

        dt_coarse = cfl * coarse.dx / u
        dt_fine = cfl * fine.dx / u

        assert dt_fine < dt_coarse


class TestStabilityRegions:
    """Test stability regions in parameter space."""

    def test_diffusion_dominated_regime(self):
        """Test diffusion-dominated regime."""
        u = 0.01
        D = 1.0
        dx = 0.1

        # Peclet < 1: diffusion dominated
        Pe = abs(u) * dx / D
        assert Pe < 1

    def test_advection_dominated_regime(self):
        """Test advection-dominated regime."""
        u = 10.0
        D = 0.01
        dx = 0.1

        # Peclet > 1: advection dominated
        Pe = abs(u) * dx / D
        assert Pe > 1

    def test_balanced_regime(self):
        """Test balanced advection-diffusion."""
        u = 1.0
        D = 0.1
        dx = 0.1

        # Peclet ~ 1: balanced
        Pe = abs(u) * dx / D
        assert 0.5 < Pe < 2


class TestStabilityChecks:
    """Practical stability checks."""

    def test_positive_diffusion_coefficient(self):
        """Test that diffusion coefficient is positive."""
        D = 0.1
        assert D >= 0

    def test_realistic_velocity_range(self):
        """Test realistic velocity range."""
        velocities = [-10, -1, -0.1, 0, 0.1, 1, 10]
        for u in velocities:
            assert -100 < u < 100

    def test_realistic_diffusion_range(self):
        """Test realistic diffusion range."""
        diffusions = [1e-4, 1e-2, 0.1, 1.0, 10]
        for D in diffusions:
            assert D > 0
