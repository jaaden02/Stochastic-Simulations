import numpy as np
from stochlib.setup import Grid
from stochlib.results import compare_distributions


def normalize_pdf(f, grid):
    return f / (np.sum(f) * grid.volume_element)


def test_compare_distributions_identical_uniform():
    grid = Grid(x_start=0.0, x_end=1.0, num_points_x=101)
    f1 = np.ones(grid.shape, dtype=float)
    f2 = np.ones(grid.shape, dtype=float)
    f1 = normalize_pdf(f1, grid)
    f2 = normalize_pdf(f2, grid)

    metrics = compare_distributions(f1, f2, grid)
    assert metrics["l1_error"] < 1e-12
    assert metrics["l2_error"] < 1e-12
    assert metrics["kl_divergence"] < 1e-12
    assert metrics["wasserstein_1d"] < 1e-12


def test_compare_distributions_shifted_mass():
    grid = Grid(x_start=0.0, x_end=1.0, num_points_x=101)
    x = grid.x_grid
    # Reference: uniform
    f_ref = normalize_pdf(np.ones_like(x), grid)
    # Paths: mass concentrated on left half
    mask_left = x <= 0.5
    f_paths = np.zeros_like(x)
    f_paths[mask_left] = 1.0
    f_paths = normalize_pdf(f_paths, grid)

    metrics = compare_distributions(f_ref, f_paths, grid)
    assert metrics["l1_error"] > 0.0
    assert metrics["l2_error"] > 0.0
    assert metrics["kl_divergence"] > 0.0
    assert metrics["wasserstein_1d"] > 0.0
