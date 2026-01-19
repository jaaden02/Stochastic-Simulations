import numpy as np
from stochlib.sde.solver import PathSimulator


def test_reflect_boundary_single_step():
    # 1D domain [0, 1]
    lower = np.array([0.0])
    upper = np.array([1.0])

    # Deterministic drift to the right, no diffusion
    def drift(x, t):
        return np.full_like(x, 0.3)

    def diffusion(x, t):
        return np.zeros_like(x)

    sim = PathSimulator(
        drift=drift,
        diffusion=diffusion,
        bounds=(lower, upper),
        boundary_mode="reflect",
        scheme="euler_maruyama",
    )

    # Start near the upper boundary; one step should cross and reflect
    x0 = np.array([0.9])
    t_array = np.array([0.0, 1.0])

    result = sim.simulate(x0=x0, t_array=t_array, n_paths=1, save_paths=True)
    x1 = result['paths'][0, 1, 0]

    # Expected reflection: x_new = 0.9 + 0.3 = 1.2 -> reflect to 2*1.0 - 1.2 = 0.8
    assert np.isclose(x1, 0.8)
    assert lower[0] <= x1 <= upper[0]
