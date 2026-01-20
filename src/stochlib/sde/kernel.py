"""Low-level stepping kernels for SDE path simulation."""

from typing import Callable, Optional
import numpy as np
from numba import njit

Array = np.ndarray


def _as_array(value, target_shape) -> Array:
    """Broadcast scalar or array-like to target shape."""
    arr = np.asarray(value, dtype=float)
    if arr.shape == target_shape:
        return arr
    return np.broadcast_to(arr, target_shape).astype(float)


@njit
def _em_step_kernel(x: Array, mu: Array, sigma: Array, dw: Array, dt: float) -> Array:
    """Core Euler-Maruyama update (numba-accelerated)."""
    result: Array = x + mu * dt + sigma * dw
    return result


@njit
def _milstein_step_kernel(
    x: Array, mu: Array, sigma: Array, sigma_x: Array, dw: Array, dt: float
) -> Array:
    """Core Milstein update (numba-accelerated)."""
    correction = 0.5 * sigma * sigma_x * (dw * dw - dt)
    result: Array = x + mu * dt + sigma * dw + correction
    return result


def euler_maruyama_step(
    x: Array,
    drift: Callable[[Array, float], Array],
    diffusion: Callable[[Array, float], Array],
    t: float,
    dt: float,
    rng: np.random.Generator,
) -> Array:
    """One Euler-Maruyama step (diagonal noise).

    Parameters
    ----------
    x : ndarray
        Current state, shape (n_paths, dim) or (dim,)
    drift : callable
        Drift function mu(x, t) returning same shape as x
    diffusion : callable
        Diffusion function sigma(x, t) returning same shape as x
    t : float
        Current time
    dt : float
        Time step
    rng : Generator
        NumPy random generator

    Returns
    -------
    ndarray
        Updated state
    """
    mu = drift(x, t)
    sigma = diffusion(x, t)
    dw = rng.normal(0.0, np.sqrt(dt), size=sigma.shape)
    result: Array = _em_step_kernel(x, mu, sigma, dw, dt)
    return result


def milstein_step(
    x: Array,
    drift: Callable[[Array, float], Array],
    diffusion: Callable[[Array, float], Array],
    diffusion_jacobian: Optional[Callable[[Array, float], Array]],
    t: float,
    dt: float,
    rng: np.random.Generator,
) -> Array:
    """One Milstein step for diagonal noise.

    If ``diffusion_jacobian`` is not provided, this falls back to Euler-Maruyama.

    Parameters
    ----------
    x : ndarray
        Current state, shape (n_paths, dim) or (dim,)
    drift : callable
        Drift function mu(x, t)
    diffusion : callable
        Diffusion function sigma(x, t)
    diffusion_jacobian : callable, optional
        Jacobian d(sigma)/dx; if None, falls back to EM
    t : float
        Current time
    dt : float
        Time step
    rng : Generator
        NumPy random generator

    Returns
    -------
    ndarray
        Updated state
    """
    mu = drift(x, t)
    sigma = diffusion(x, t)
    dw = rng.normal(0.0, np.sqrt(dt), size=sigma.shape)

    if diffusion_jacobian is None:
        result: Array = _em_step_kernel(x, mu, sigma, dw, dt)
        return result

    sigma_x = diffusion_jacobian(x, t)
    result_mil: Array = _milstein_step_kernel(x, mu, sigma, sigma_x, dw, dt)
    return result_mil


def normalize_callable(spec, dim: int) -> Callable[[Array, float], Array]:
    """Return a callable drift/diffusion from scalar, array, or function.

    Parameters
    ----------
    spec : callable, scalar, or array
        Function, constant, or array specification
    dim : int
        State dimension

    Returns
    -------
    callable
        Function accepting (x, t) and returning array
    """
    if callable(spec):
        result_fn: Callable[[Array, float], Array] = spec
        return result_fn
    # Constant drift/diffusion
    const = np.asarray(spec, dtype=float)
    if const.shape == ():
        const = np.full((dim,), float(const))

    def _fn(x: Array, t: float) -> Array:
        shape = x.shape
        target_shape = shape if shape[-1] == dim else (shape[0], dim)
        return _as_array(const, target_shape)

    return _fn

    return _fn
