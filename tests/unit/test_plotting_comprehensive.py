"""Comprehensive tests for plotting modules across all solvers."""

import numpy as np
import matplotlib.pyplot as plt


class TestSDEPlottingImports:
    """Tests that SDE plotting modules can be imported."""

    def test_sde_paths_module_imports(self):
        """Test that SDE paths module imports."""
        from stochlib.sde.plotting import paths as sde_paths

        assert hasattr(sde_paths, "plot_paths_with_threshold")

    def test_sde_trajectories_module_imports(self):
        """Test that SDE trajectories module imports."""
        from stochlib.sde.plotting import trajectories

        assert hasattr(trajectories, "plot_trajectories")

    def test_sde_phase_portrait_module_imports(self):
        """Test that SDE phase portrait module imports."""
        from stochlib.sde.plotting import phase_portrait

        assert hasattr(phase_portrait, "plot_phase_portrait")

    def test_sde_plotting_init_has_exports(self):
        """Test that SDE plotting __init__ exports functions."""
        from stochlib import sde

        assert hasattr(sde.plotting, "plot_trajectories")
        assert hasattr(sde.plotting, "plot_phase_portrait")


class TestDeterministicPlottingImports:
    """Tests that Deterministic plotting modules can be imported."""

    def test_deterministic_evolution_module_imports(self):
        """Test evolution plotting module imports."""
        from stochlib.deterministic.plotting import evolution

        assert hasattr(evolution, "plot_solution_evolution")

    def test_deterministic_comparison_module_imports(self):
        """Test comparison plotting module imports."""
        from stochlib.deterministic.plotting import comparison

        assert hasattr(comparison, "plot_comparison_with_exact")

    def test_deterministic_diagnostic_module_imports(self):
        """Test diagnostic plotting module imports."""
        from stochlib.deterministic.plotting import diagnostic_plots

        assert hasattr(diagnostic_plots, "plot_diagnostics")

    def test_deterministic_plotting_init_has_exports(self):
        """Test deterministic plotting __init__ exports functions."""
        from stochlib import deterministic

        assert hasattr(deterministic.plotting, "plot_solution_evolution")


class TestFokkerPlanckPlottingImports:
    """Tests that Fokker-Planck plotting modules can be imported."""

    def test_fokker_planck_distributions_module_imports(self):
        """Test distributions module imports."""
        from stochlib.fokker_planck.plotting import distributions

        assert hasattr(distributions, "plot_1d_distribution")

    def test_fokker_planck_animation_module_imports(self):
        """Test animation module imports."""
        from stochlib.fokker_planck.plotting import animation

        # Animation module has utility functions, check it imports without error
        assert animation is not None

    def test_fokker_planck_diagnostics_module_imports(self):
        """Test diagnostics plotting module imports."""
        from stochlib.fokker_planck.plotting import diagnostics

        assert hasattr(diagnostics, "plot_mass_conservation")

    def test_fokker_planck_plotting_init_has_exports(self):
        """Test FP plotting __init__ exports functions."""
        from stochlib import fokker_planck

        assert hasattr(fokker_planck.plotting, "plot_1d_distribution")


class TestPlottingBasicFunctionality:
    """Smoke tests that plotting functions can be called."""

    def test_sde_trajectories_callable(self, simple_1d_grid):
        """Test that SDE trajectory plotting function is callable."""
        from stochlib.sde.plotting import plot_trajectories

        # Create dummy path data
        paths = np.random.randn(10, 11, 1)  # 10 paths, 11 time steps, 1 dimension
        times = np.linspace(0, 1, 11)

        fig, ax = plt.subplots()
        try:
            result_ax = plot_trajectories(
                paths=paths,
                times=times,
                ax=ax,
                max_paths=5,
            )
            assert result_ax is not None
        finally:
            plt.close("all")

    def test_deterministic_evolution_callable(self, simple_1d_grid):
        """Test that deterministic evolution plotting is callable."""
        from stochlib.deterministic.plotting import plot_solution_evolution

        # Create dummy solution data
        t_eval = np.linspace(0, 1, 6)
        u = np.random.rand(6, simple_1d_grid.num_points_x)
        x = simple_1d_grid.x_grid

        fig, ax = plt.subplots()
        try:
            result_ax = plot_solution_evolution(
                t_eval=t_eval,
                u=u,
                x=x,
                ax=ax,
                n_snapshots=3,
            )
            assert result_ax is not None
        finally:
            plt.close("all")

    def test_fokker_planck_distribution_callable(self, simple_1d_grid):
        """Test that FP distribution plotting is callable."""
        from stochlib.fokker_planck.plotting import plot_1d_distribution

        # Create dummy distribution
        f = np.exp(-((simple_1d_grid.x_grid - 5.0) ** 2) / (2 * 0.5**2))
        f = f / np.trapz(f, dx=simple_1d_grid.dx)

        fig, ax = plt.subplots()
        try:
            result_ax = plot_1d_distribution(
                grid=simple_1d_grid,
                f=f,
                ax=ax,
                label="Test distribution",
            )
            assert result_ax is not None
        finally:
            plt.close("all")
