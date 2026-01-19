import numpy as np
from stochlib.setup import Grid, InitialCondition
from stochlib.sde.sampling import sample_from_ic


def test_sample_from_ic_gaussian_mean_and_bounds():
    rng = np.random.default_rng(42)
    grid = Grid(x_start=0.0, x_end=10.0, num_points_x=101)
    ic = InitialCondition(grid, func_type='gaussian', x0=5.0, sigma_x=0.5)

    samples = sample_from_ic(ic, n_samples=2000, rng=rng)
    assert samples.shape == (2000, 1)

    # Samples stay within grid bounds
    assert np.all(samples[:, 0] >= grid.x_start)
    assert np.all(samples[:, 0] <= grid.x_end)

    # Mean is close to the specified center
    mean = np.mean(samples[:, 0])
    assert abs(mean - 5.0) < 0.2
