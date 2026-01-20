"""Sampling utilities for initial conditions."""

import numpy as np
from typing import Optional
from ..setup import Grid, InitialCondition

Array = np.ndarray


def sample_from_distribution(
    f: Array,
    grid: Grid,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> Array:
    """Sample initial positions from a probability distribution on a grid.

    Parameters
    ----------
    f : ndarray
        Probability distribution matching grid.shape, need not be normalized
    grid : Grid
        Spatial grid defining the domain
    n_samples : int
        Number of samples to draw
    rng : Generator, optional
        Random number generator

    Returns
    -------
    ndarray
        Sampled positions, shape (n_samples, dim)

    Notes
    -----
    Uses inverse transform sampling on the flattened distribution.
    """
    rng = rng or np.random.default_rng()

    # Normalize distribution
    f_flat = f.flatten()
    f_flat = f_flat / np.sum(f_flat)

    # Sample flat indices
    flat_indices = rng.choice(len(f_flat), size=n_samples, p=f_flat)

    # Convert to multi-dimensional indices
    multi_indices = np.unravel_index(flat_indices, grid.shape)

    # Convert indices to continuous positions
    dim = len(grid.axis_names)
    samples = np.zeros((n_samples, dim))

    for i, ax in enumerate(grid.axis_names):
        if ax == "x":
            grid_ax = grid.x_grid
        elif ax == "y":
            grid_ax = grid.y_grid
        elif ax == "z":
            grid_ax = grid.z_grid
        else:
            raise ValueError(
                f"Unknown axis '{ax}'. Valid axes for grid sampling: 'x', 'y', 'z'. "
                f"Check your grid axes and ensure sampling axis matches grid dimensions."
            )

        # Map grid index to position
        samples[:, i] = grid_ax[multi_indices[i]]

    return samples


def sample_from_ic(
    ic: InitialCondition,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> Array:
    """Sample initial positions from an InitialCondition object.

    Parameters
    ----------
    ic : InitialCondition
        Initial condition with distribution f0 and grid
    n_samples : int
        Number of samples to draw
    rng : Generator, optional
        Random number generator

    Returns
    -------
    ndarray
        Sampled positions, shape (n_samples, dim)
    """
    return sample_from_distribution(ic.f0, ic.grid, n_samples, rng=rng)
