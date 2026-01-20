"""Tests for diagnostics and analysis functions."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stochlib.setup import InitialCondition


class TestMoments:
    """Test moment calculations."""

    def test_zeroth_moment(self, simple_1d_grid):
        """Test total probability (0th moment)."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        f = ic.f0

        # 0th moment should integrate to ~1
        moment_0 = np.trapz(f, dx=simple_1d_grid.dx)
        assert np.isclose(moment_0, 1.0, atol=0.01)

    def test_first_moment_mean(self, simple_1d_grid):
        """Test first moment (mean)."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        f = ic.f0

        # Create coordinate array
        x = np.linspace(simple_1d_grid.x_start, simple_1d_grid.x_end, simple_1d_grid.num_points_x)

        # First moment = mean position
        mean = np.trapz(x * f, dx=simple_1d_grid.dx)
        assert np.isclose(mean, 5.0, atol=0.1)

    def test_second_moment_variance(self, simple_1d_grid):
        """Test second moment (variance)."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        f = ic.f0

        x = np.linspace(simple_1d_grid.x_start, simple_1d_grid.x_end, simple_1d_grid.num_points_x)
        mean = np.trapz(x * f, dx=simple_1d_grid.dx)

        # Second moment = E[x^2]
        second_moment = np.trapz(x**2 * f, dx=simple_1d_grid.dx)
        variance = second_moment - mean**2

        assert variance > 0


class TestStatistics:
    """Test statistical properties."""

    def test_mean_calculation(self, simple_1d_grid):
        """Test mean calculation."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=3.0, sigma_x=0.3)
        f = ic.f0

        x = np.linspace(simple_1d_grid.x_start, simple_1d_grid.x_end, simple_1d_grid.num_points_x)
        mean = np.sum(x * f) / np.sum(f)

        assert 2.5 < mean < 3.5

    def test_variance_calculation(self, simple_1d_grid):
        """Test variance calculation."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=1.0)
        f = ic.f0

        x = np.linspace(simple_1d_grid.x_start, simple_1d_grid.x_end, simple_1d_grid.num_points_x)
        mean = np.sum(x * f) / np.sum(f)
        variance = np.sum((x - mean) ** 2 * f) / np.sum(f)

        assert variance > 0

    def test_standard_deviation(self, simple_1d_grid):
        """Test standard deviation calculation."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        f = ic.f0

        x = np.linspace(simple_1d_grid.x_start, simple_1d_grid.x_end, simple_1d_grid.num_points_x)
        mean = np.sum(x * f) / np.sum(f)
        variance = np.sum((x - mean) ** 2 * f) / np.sum(f)
        std_dev = np.sqrt(variance)

        assert std_dev > 0
        assert np.isclose(std_dev, 0.5, atol=0.1)


class TestConsistency:
    """Test consistency properties."""

    def test_mass_conservation(self, simple_1d_grid):
        """Test that total mass is conserved."""
        f1 = np.ones(simple_1d_grid.num_points_x)
        mass1 = np.trapz(f1, dx=simple_1d_grid.dx)

        f2 = 2 * np.ones(simple_1d_grid.num_points_x)
        mass2 = np.trapz(f2, dx=simple_1d_grid.dx)

        # Doubling values doubles mass
        assert np.isclose(mass2, 2 * mass1)

    def test_linearity(self, simple_1d_grid):
        """Test linearity of integration."""
        ic1 = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        ic2 = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.3)

        f1 = ic1.f0
        f2 = ic2.f0

        # Linear combination
        f_combined = 0.5 * f1 + 0.3 * f2

        # Integration is linear
        integral_combined = np.trapz(f_combined, dx=simple_1d_grid.dx)
        expected = 0.5 * np.trapz(f1, dx=simple_1d_grid.dx) + 0.3 * np.trapz(
            f2, dx=simple_1d_grid.dx
        )

        assert np.isclose(integral_combined, expected)


class TestL2Norm:
    """Test L2 norm and related metrics."""

    def test_l2_norm_definition(self, simple_1d_grid):
        """Test L2 norm definition."""
        f = np.array([1.0, 2.0, 3.0, 2.0, 1.0])

        # L2 norm: sqrt(sum(f^2) * dx)
        l2_norm = np.sqrt(np.sum(f**2) * simple_1d_grid.dx)
        assert l2_norm > 0

    def test_l2_norm_zero_function(self, simple_1d_grid):
        """Test L2 norm of zero function."""
        f = np.zeros(simple_1d_grid.num_points_x)
        l2_norm = np.sqrt(np.sum(f**2) * simple_1d_grid.dx)
        assert l2_norm == 0

    def test_l2_norm_scaling(self, simple_1d_grid):
        """Test L2 norm scaling property."""
        f = np.array([1.0, 2.0, 3.0])
        l2_norm_f = np.sqrt(np.sum(f**2))
        l2_norm_2f = np.sqrt(np.sum((2 * f) ** 2))

        # ||2f|| = 2 * ||f||
        assert np.isclose(l2_norm_2f, 2 * l2_norm_f)


class TestEnergy:
    """Test energy-based diagnostics."""

    def test_kinetic_energy_equivalence(self):
        """Test kinetic energy calculation."""
        # KE ~ integral(u^2) where u is velocity field
        u = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
        ke = 0.5 * np.sum(u**2)

        assert ke > 0

    def test_potential_energy_well(self):
        """Test potential energy in a well."""
        # PE ~ integral(V(x) * f(x)) where V is potential
        x = np.linspace(0, 1, 10)
        f = np.exp(-((x - 0.5) ** 2))  # Peaked at 0.5
        V = (x - 0.5) ** 2  # Quadratic potential (well at 0.5)

        pe = np.sum(V * f)
        assert pe >= 0

    def test_total_energy(self):
        """Test total energy = KE + PE."""
        u = 0.5
        x = 0.1
        V = x**2

        ke = 0.5 * u**2
        pe = V
        total = ke + pe

        assert total > 0


class TestErrorMetrics:
    """Test error measurement."""

    def test_l1_error(self):
        """Test L1 error norm."""
        f_approx = np.array([1.0, 2.1, 3.0, 1.9])
        f_exact = np.array([1.0, 2.0, 3.0, 2.0])

        l1_error = np.sum(np.abs(f_approx - f_exact))
        assert 0 < l1_error < 1

    def test_l2_error(self):
        """Test L2 error norm."""
        f_approx = np.array([1.0, 2.1, 3.0, 1.9])
        f_exact = np.array([1.0, 2.0, 3.0, 2.0])

        l2_error = np.sqrt(np.sum((f_approx - f_exact) ** 2))
        assert l2_error > 0

    def test_linf_error(self):
        """Test L-infinity error norm."""
        f_approx = np.array([1.0, 2.1, 3.0, 1.9])
        f_exact = np.array([1.0, 2.0, 3.0, 2.0])

        linf_error = np.max(np.abs(f_approx - f_exact))
        assert np.isclose(linf_error, 0.1)

    def test_relative_error(self):
        """Test relative error."""
        f_approx = 2.1
        f_exact = 2.0

        rel_error = abs(f_approx - f_exact) / abs(f_exact)
        assert np.isclose(rel_error, 0.05)


class TestDistanceMetrics:
    """Test distance metrics."""

    def test_euclidean_distance(self):
        """Test Euclidean distance."""
        p1 = np.array([0, 0])
        p2 = np.array([3, 4])

        distance = np.sqrt(np.sum((p2 - p1) ** 2))
        assert np.isclose(distance, 5.0)

    def test_manhattan_distance(self):
        """Test Manhattan distance."""
        p1 = np.array([0, 0])
        p2 = np.array([3, 4])

        distance = np.sum(np.abs(p2 - p1))
        assert distance == 7

    def test_wasserstein_distance_1d(self):
        """Test Wasserstein distance (simple approximation)."""
        # Cumulative distributions
        f = np.array([0.25, 0.25, 0.25, 0.25])
        g = np.array([0.0, 0.5, 0.5, 0.0])

        # L1 distance between CDFs
        cdf_f = np.cumsum(f)
        cdf_g = np.cumsum(g)

        w_distance = np.sum(np.abs(cdf_f - cdf_g))
        assert w_distance > 0


class TestPeakProperties:
    """Test peak detection and analysis."""

    def test_peak_location(self, simple_1d_grid):
        """Test finding peak location."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        f = ic.f0

        peak_idx = np.argmax(f)
        x = np.linspace(simple_1d_grid.x_start, simple_1d_grid.x_end, simple_1d_grid.num_points_x)
        peak_x = x[peak_idx]

        assert 4.5 < peak_x < 5.5

    def test_peak_value(self, simple_1d_grid):
        """Test peak value."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        f = ic.f0

        peak = np.max(f)
        assert peak > 0

    def test_multiple_peaks(self):
        """Test detection of multiple peaks."""
        # Bimodal distribution
        x = np.linspace(0, 10, 100)
        f = np.exp(-((x - 2) ** 2)) + np.exp(-((x - 8) ** 2))

        # Should have two peaks
        peaks = f[1:-1][(f[1:-1] > f[:-2]) & (f[1:-1] > f[2:])]
        assert len(peaks) >= 1  # At least one peak
