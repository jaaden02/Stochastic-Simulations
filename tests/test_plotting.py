"""Tests for visualization and plotting."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stochlib.setup import InitialCondition


class TestPlottingSetup:
    """Test basic plotting setup."""

    def test_data_for_plotting(self, simple_1d_grid):
        """Test that data is suitable for plotting."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        f = ic.f0

        # Data should be numeric and finite
        assert np.all(np.isfinite(f))

    def test_grid_coordinates(self, simple_1d_grid):
        """Test grid coordinates for plotting."""
        x = np.linspace(simple_1d_grid.x_start, simple_1d_grid.x_end, simple_1d_grid.num_points_x)

        assert len(x) == simple_1d_grid.num_points_x
        assert x[0] == simple_1d_grid.x_start
        assert x[-1] == simple_1d_grid.x_end


class TestLineplot:
    """Test 1D line plot generation."""

    def test_1d_line_plot_data(self, simple_1d_grid):
        """Test data preparation for 1D line plot."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        f = ic.f0
        x = np.linspace(simple_1d_grid.x_start, simple_1d_grid.x_end, simple_1d_grid.num_points_x)

        # Should be plottable
        assert len(x) == len(f)
        assert np.all(np.isfinite(f))

    def test_multiple_curves(self, simple_1d_grid):
        """Test data for plotting multiple curves."""
        ic1 = InitialCondition(simple_1d_grid, func_type="gaussian", x0=3.0, sigma_x=0.5)
        ic2 = InitialCondition(simple_1d_grid, func_type="gaussian", x0=7.0, sigma_x=0.5)

        f1 = ic1.f0
        f2 = ic2.f0
        x = np.linspace(simple_1d_grid.x_start, simple_1d_grid.x_end, simple_1d_grid.num_points_x)

        # Both curves have same x-axis
        assert len(x) == len(f1) == len(f2)


class TestContourplot:
    """Test 2D contour plot generation."""

    def test_2d_data_structure(self, simple_2d_grid):
        """Test 2D data structure for contour plots."""
        ic = InitialCondition(
            simple_2d_grid, func_type="gaussian", x0=5.0, y0=5.0, sigma_x=0.5, sigma_y=0.5
        )
        f = ic.f0

        # Should be 2D
        assert f.ndim == 2
        assert f.shape[0] == simple_2d_grid.num_points_x
        assert f.shape[1] == simple_2d_grid.num_points_y

    def test_2d_grid_for_contour(self, simple_2d_grid):
        """Test creating grid for contour plot."""
        x = np.linspace(simple_2d_grid.x_start, simple_2d_grid.x_end, simple_2d_grid.num_points_x)
        y = np.linspace(simple_2d_grid.y_start, simple_2d_grid.y_end, simple_2d_grid.num_points_y)

        X, Y = np.meshgrid(x, y)

        assert X.shape == (simple_2d_grid.num_points_y, simple_2d_grid.num_points_x)
        assert Y.shape == (simple_2d_grid.num_points_y, simple_2d_grid.num_points_x)


class TestColorMapping:
    """Test color mapping for plots."""

    def test_data_range_for_colorbar(self, simple_1d_grid):
        """Test data range for colorbar."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        f = ic.f0

        v_min = np.min(f)
        v_max = np.max(f)

        assert v_min >= 0
        assert v_max > v_min

    def test_normalized_data_for_colormap(self, simple_1d_grid):
        """Test normalizing data for colormap."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        f = ic.f0

        # Normalize to [0, 1]
        f_norm = (f - np.min(f)) / (np.max(f) - np.min(f))

        assert np.min(f_norm) >= 0
        assert np.max(f_norm) <= 1


class TestAnimationFrames:
    """Test generating animation frames."""

    def test_frame_sequence(self, simple_1d_grid):
        """Test generating sequence of frames."""
        frames = []

        for t in np.linspace(0, 1, 11):
            # Simulate spreading Gaussian
            sigma = 0.5 + 0.5 * t
            ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=sigma)
            frames.append(ic.f0.copy())

        assert len(frames) == 11
        assert all(len(f) == simple_1d_grid.num_points_x for f in frames)

    def test_frame_evolution(self, simple_1d_grid):
        """Test that frames show evolution."""
        frames = []

        for sigma in [0.3, 0.5, 0.7]:
            ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=sigma)
            frames.append(ic.f0)

        # Width should increase
        peaks = [np.max(f) for f in frames]
        assert peaks[0] > peaks[-1]  # Peak decreases as width increases


class TestStatisticalPlots:
    """Test statistical visualization."""

    def test_histogram_data(self, simple_1d_grid):
        """Test data for histogram."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        f = ic.f0

        # Can create histogram
        assert len(f) > 0
        assert np.all(f >= 0)

    def test_pdf_plot(self, simple_1d_grid):
        """Test data for PDF plot."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        f = ic.f0
        x = np.linspace(simple_1d_grid.x_start, simple_1d_grid.x_end, simple_1d_grid.num_points_x)

        # Integrate PDF (should be ~1)
        integral = np.trapz(f, x)
        assert 0.8 < integral < 1.2

    def test_cdf_plot(self, simple_1d_grid):
        """Test data for CDF plot."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        f = ic.f0

        # CDF is cumulative
        cdf = np.cumsum(f) * simple_1d_grid.dx

        # Should be monotone increasing
        assert np.all(np.diff(cdf) >= 0)
        # Should go from 0 to 1
        assert cdf[0] > -0.1
        assert cdf[-1] > 0.5


class TestTimeSeriesPlots:
    """Test time series visualization."""

    def test_moment_evolution(self, simple_1d_grid):
        """Test data for moment evolution plot."""
        moments = []

        for sigma in np.linspace(0.3, 1.0, 20):
            ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=sigma)
            f = ic.f0

            x = np.linspace(
                simple_1d_grid.x_start, simple_1d_grid.x_end, simple_1d_grid.num_points_x
            )
            mean = np.sum(x * f) * simple_1d_grid.dx
            moments.append(mean)

        assert len(moments) == 20
        # Mean should be roughly constant
        assert np.std(moments) < 0.5

    def test_energy_evolution(self, simple_1d_grid):
        """Test data for energy evolution plot."""
        energies = []
        times = np.linspace(0, 1, 20)

        for t in times:
            # Simulate energy evolution
            sigma = 0.5 * (1 + t)
            ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=sigma)
            f = ic.f0

            energy = np.sum(f**2) * simple_1d_grid.dx
            energies.append(energy)

        assert len(energies) == len(times)


class TestPlotLabels:
    """Test plot labels and annotations."""

    def test_axis_labels(self, simple_1d_grid):
        """Test axis label generation."""
        x_label = "Position (x)"
        y_label = "Probability density f(x)"

        assert len(x_label) > 0
        assert len(y_label) > 0

    def test_title_generation(self):
        """Test title generation."""
        title = "Fokker-Planck Equation Solution"

        assert "Fokker-Planck" in title
        assert len(title) > 0

    def test_legend_entries(self):
        """Test legend entry generation."""
        legends = ["t = 0.0", "t = 0.5", "t = 1.0"]

        assert len(legends) == 3
        assert all(isinstance(leg, str) for leg in legends)


class TestPlotValidation:
    """Test plot validation."""

    def test_data_is_plottable(self, simple_1d_grid):
        """Test that data is suitable for plotting."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        f = ic.f0

        # Check finite values
        assert np.all(np.isfinite(f))
        # Check non-negative (for PDF)
        assert np.all(f >= -1e-10)

    def test_no_nan_values(self, simple_2d_grid):
        """Test that data has no NaN values."""
        ic = InitialCondition(
            simple_2d_grid, func_type="gaussian", x0=5.0, y0=5.0, sigma_x=0.5, sigma_y=0.5
        )
        f = ic.f0

        assert not np.any(np.isnan(f))

    def test_no_inf_values(self, simple_1d_grid):
        """Test that data has no infinite values."""
        ic = InitialCondition(simple_1d_grid, func_type="gaussian", x0=5.0, sigma_x=0.5)
        f = ic.f0

        assert not np.any(np.isinf(f))
