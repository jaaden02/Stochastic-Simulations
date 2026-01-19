"""Bridge utilities to convert Fokker-Planck configs to SDE callables."""

from typing import Callable, Tuple
import numpy as np
from ..setup import Grid, VelocitiesConfig, DiffusionConfig
from ..logging_utils import get_logger

logger = get_logger("sde.bridge")
Array = np.ndarray


def fp_to_sde_drift(
    velocities: VelocitiesConfig,
    grid: Grid,
) -> Callable[[Array, float], Array]:
    """Convert VelocitiesConfig to SDE drift callable.

    Parameters
    ----------
    velocities : VelocitiesConfig
        Fokker-Planck velocity configuration
    grid : Grid
        Spatial grid (used to determine dimensionality)

    Returns
    -------
    callable
        Drift function mu(x, t) returning array of shape x.shape
    """
    axis_names = grid.axis_names
    dim = len(axis_names)

    def drift(x: Array, t: float) -> Array:
        # x can be (n_paths, dim) or (dim,)
        single_path = x.ndim == 1
        if single_path:
            x = x[None, :]  # (1, dim)

        n_paths = x.shape[0]
        mu = np.zeros_like(x)

        # Build velocity fields at time t
        vel_fields = velocities.build_fields(t=t, evaluate=False)

        for i, ax in enumerate(axis_names):
            vel_spec = vel_fields.get(ax)
            if vel_spec is None:
                continue

            if callable(vel_spec):
                # Evaluate at each path position
                for path_idx in range(n_paths):
                    # Build coordinate tuple for this path
                    coords = []
                    for j, ax_name in enumerate(axis_names):
                        coords.append(x[path_idx, j])
                    try:
                        mu[path_idx, i] = vel_spec(*coords, t)
                    except TypeError:
                        mu[path_idx, i] = vel_spec(*coords)
            else:
                # Constant velocity
                mu[:, i] = float(vel_spec)

        if single_path:
            return mu[0, :]
        return mu

    return drift


def fp_to_sde_diffusion(
    diffusions: DiffusionConfig,
    grid: Grid,
) -> Callable[[Array, float], Array]:
    """Convert DiffusionConfig to SDE diffusion callable (diagonal).

    Parameters
    ----------
    diffusions : DiffusionConfig
        Fokker-Planck diffusion configuration
    grid : Grid
        Spatial grid

    Returns
    -------
    callable
        Diffusion function sigma(x, t) returning array of shape x.shape

    Notes
    -----
    Returns sqrt(2*D) since SDE uses dX = mu*dt + sigma*dW while
    FP uses d/dx(D*df/dx) corresponding to variance 2*D.
    """
    axis_names = grid.axis_names
    dim = len(axis_names)

    def diffusion(x: Array, t: float) -> Array:
        single_path = x.ndim == 1
        if single_path:
            x = x[None, :]

        n_paths = x.shape[0]
        sigma = np.zeros_like(x)

        # Build diffusion fields at time t
        diff_fields = diffusions.build_fields(t=t, evaluate=False)

        for i, ax in enumerate(axis_names):
            diff_spec = diff_fields.get(ax)
            if diff_spec is None:
                continue

            if callable(diff_spec):
                for path_idx in range(n_paths):
                    coords = []
                    for j, ax_name in enumerate(axis_names):
                        coords.append(x[path_idx, j])
                    try:
                        D_val = diff_spec(*coords, t)
                    except TypeError:
                        D_val = diff_spec(*coords)
                    sigma[path_idx, i] = np.sqrt(2.0 * D_val)
            else:
                # Constant diffusion
                D_val = float(diff_spec)
                sigma[:, i] = np.sqrt(2.0 * D_val)

        if single_path:
            return sigma[0, :]
        return sigma

    return diffusion


def grid_bounds_from_grid(grid: Grid) -> Tuple[Array, Array]:
    """Extract boundary bounds from Grid.

    Parameters
    ----------
    grid : Grid
        Spatial grid

    Returns
    -------
    lower : ndarray
        Lower bounds for each dimension
    upper : ndarray
        Upper bounds for each dimension
    """
    lower = []
    upper = []

    if "x" in grid.axis_names:
        lower.append(grid.x_start)
        upper.append(grid.x_end)
    if "y" in grid.axis_names:
        lower.append(grid.y_start)
        upper.append(grid.y_end)
    if "z" in grid.axis_names:
        lower.append(grid.z_start)
        upper.append(grid.z_end)

    return np.array(lower), np.array(upper)
