import numpy as np
from stochlib.setup import Grid, VelocitiesConfig, DiffusionConfig
from stochlib.sde.bridge import fp_to_sde_drift, fp_to_sde_diffusion


def test_fp_to_sde_drift_constant():
    grid = Grid(x_start=0.0, x_end=1.0, num_points_x=16)
    velocities = VelocitiesConfig(grid, mu_x=1.5)
    drift = fp_to_sde_drift(velocities, grid)

    x = np.array([[0.2], [0.8]], dtype=float)
    mu = drift(x, t=0.0)

    assert mu.shape == x.shape
    assert np.allclose(mu[:, 0], 1.5)


def test_fp_to_sde_diffusion_constant():
    grid = Grid(x_start=0.0, x_end=1.0, num_points_x=16)
    diffusions = DiffusionConfig(grid, axes=["x"], constants={"x": 0.5})
    diffusion = fp_to_sde_diffusion(diffusions, grid)

    x = np.array([[0.2], [0.8]], dtype=float)
    sigma = diffusion(x, t=0.0)

    assert sigma.shape == x.shape
    # sqrt(2*D) with D=0.5 -> 1.0
    assert np.allclose(sigma[:, 0], 1.0)
