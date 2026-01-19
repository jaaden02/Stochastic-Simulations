"""Diagnostic tools for deterministic PDE solutions.

This module provides analysis functions for evaluating deterministic PDE solutions,
including mass conservation, error metrics, stability checks, and solution quality.
"""
from typing import Optional
import numpy as np
from ..logging_utils import get_logger

logger = get_logger("deterministic.diagnostics")


def check_mass_conservation(
    u: np.ndarray,
    dx: float,
    u_initial: Optional[np.ndarray] = None,
    tolerance: float = 1e-6,
) -> dict:
    """Check mass conservation in deterministic solution.
    
    Parameters
    ----------
    u : ndarray
        Current solution (n_grid,)
    dx : float
        Grid spacing
    u_initial : ndarray, optional
        Initial solution for comparison
    tolerance : float
        Tolerance for conservation check
        
    Returns
    -------
    dict : Conservation metrics
        - mass: Current mass
        - mass_initial: Initial mass (if provided)
        - mass_error: Absolute error in mass conservation
        - relative_error: Relative error
        - conserved: Boolean indicating if mass is conserved within tolerance
    """
    mass = np.sum(u) * dx
    
    result = {"mass": mass}
    
    if u_initial is not None:
        mass_initial = np.sum(u_initial) * dx
        mass_error = abs(mass - mass_initial)
        relative_error = mass_error / mass_initial if mass_initial != 0 else 0.0
        
        result.update({
            "mass_initial": mass_initial,
            "mass_error": mass_error,
            "relative_error": relative_error,
            "conserved": mass_error < tolerance,
        })
    
    return result


def compute_cfl_number(
    v: np.ndarray,
    dt: float,
    dx: float,
) -> dict:
    """Compute CFL (Courant-Friedrichs-Lewy) number for stability analysis.
    
    CFL = |v| * dt / dx
    
    For stability: CFL ≤ 1 (for most explicit schemes)
    
    Parameters
    ----------
    v : ndarray
        Velocity field
    dt : float
        Time step
    dx : float
        Grid spacing
        
    Returns
    -------
    dict : CFL metrics
        - cfl_max: Maximum CFL number
        - cfl_mean: Mean CFL number
        - stable: Whether CFL condition is satisfied
        - margin: Safety margin (1.0 - cfl_max)
    """
    cfl = np.abs(v) * dt / dx
    cfl_max = np.max(cfl)
    cfl_mean = np.mean(cfl)
    
    return {
        "cfl_max": cfl_max,
        "cfl_mean": cfl_mean,
        "stable": cfl_max <= 1.0,
        "margin": 1.0 - cfl_max,
    }


def compute_solution_moments(
    u: np.ndarray,
    x: np.ndarray,
    dx: float,
) -> dict:
    """Compute statistical moments of the solution.
    
    Parameters
    ----------
    u : ndarray
        Solution (n_grid,)
    x : ndarray
        Grid points (n_grid,)
    dx : float
        Grid spacing
        
    Returns
    -------
    dict : Moments
        - mass: Total mass (0th moment)
        - mean: Mean position (1st moment)
        - variance: Variance (2nd central moment)
        - std: Standard deviation
        - skewness: Skewness (3rd standardized moment)
    """
    mass = np.sum(u) * dx
    
    if mass == 0:
        return {
            "mass": 0.0,
            "mean": 0.0,
            "variance": 0.0,
            "std": 0.0,
            "skewness": 0.0,
        }
    
    # Normalize to probability distribution
    p = u / mass
    
    # Moments
    mean = np.sum(x * p) * dx
    variance = np.sum((x - mean)**2 * p) * dx
    std = np.sqrt(variance)
    
    if std > 0:
        skewness = np.sum(((x - mean) / std)**3 * p) * dx
    else:
        skewness = 0.0
    
    return {
        "mass": mass,
        "mean": mean,
        "variance": variance,
        "std": std,
        "skewness": skewness,
    }


def compute_l2_norm(u: np.ndarray, dx: float) -> float:
    """Compute L2 norm of solution.
    
    Parameters
    ----------
    u : ndarray
        Solution
    dx : float
        Grid spacing
        
    Returns
    -------
    float : L2 norm
    """
    return np.sqrt(np.sum(u**2) * dx)


def compute_error_metrics(
    u_numeric: np.ndarray,
    u_exact: np.ndarray,
    dx: float,
) -> dict:
    """Compute error metrics between numerical and exact solutions.
    
    Parameters
    ----------
    u_numeric : ndarray
        Numerical solution
    u_exact : ndarray
        Exact/reference solution
    dx : float
        Grid spacing
        
    Returns
    -------
    dict : Error metrics
        - l1_error: L1 norm of error
        - l2_error: L2 norm of error
        - linf_error: L-infinity norm of error
        - relative_l2: Relative L2 error
    """
    error = u_numeric - u_exact
    
    l1_error = np.sum(np.abs(error)) * dx
    l2_error = np.sqrt(np.sum(error**2) * dx)
    linf_error = np.max(np.abs(error))
    
    u_exact_norm = compute_l2_norm(u_exact, dx)
    relative_l2 = l2_error / u_exact_norm if u_exact_norm > 0 else 0.0
    
    return {
        "l1_error": l1_error,
        "l2_error": l2_error,
        "linf_error": linf_error,
        "relative_l2": relative_l2,
    }


def check_positivity(u: np.ndarray, tolerance: float = 1e-10) -> dict:
    """Check if solution maintains positivity (physical constraint).
    
    Parameters
    ----------
    u : ndarray
        Solution
    tolerance : float
        Tolerance for negative values
        
    Returns
    -------
    dict : Positivity check results
        - all_positive: Boolean indicating all values are non-negative
        - min_value: Minimum value in solution
        - num_negative: Number of negative values
        - negative_locations: Indices of negative values
    """
    negative_mask = u < -tolerance
    
    return {
        "all_positive": not np.any(negative_mask),
        "min_value": np.min(u),
        "num_negative": np.sum(negative_mask),
        "negative_locations": np.where(negative_mask)[0],
    }


def compute_total_variation(u: np.ndarray) -> float:
    """Compute total variation of solution.
    
    TV(u) = sum|u[i+1] - u[i]|
    
    Total variation diminishing (TVD) schemes satisfy: TV(u_new) ≤ TV(u_old)
    
    Parameters
    ----------
    u : ndarray
        Solution
        
    Returns
    -------
    float : Total variation
    """
    return np.sum(np.abs(np.diff(u)))


class DeterministicDiagnostics:
    """Diagnostic tracker for deterministic PDE solutions.
    
    Tracks solution quality, conservation, and stability metrics over time.
    
    Attributes
    ----------
    history : dict
        Dictionary storing diagnostic metrics over time
    """
    
    def __init__(self):
        """Initialize diagnostic tracker."""
        self.history = {
            "time": [],
            "mass": [],
            "mean": [],
            "variance": [],
            "l2_norm": [],
            "total_variation": [],
            "min_value": [],
            "max_value": [],
        }
    
    def update(
        self,
        t: float,
        u: np.ndarray,
        x: np.ndarray,
        dx: float,
    ):
        """Update diagnostics with current solution.
        
        Parameters
        ----------
        t : float
            Current time
        u : ndarray
            Current solution
        x : ndarray
            Grid points
        dx : float
            Grid spacing
        """
        moments = compute_solution_moments(u, x, dx)
        l2 = compute_l2_norm(u, dx)
        tv = compute_total_variation(u)
        
        self.history["time"].append(t)
        self.history["mass"].append(moments["mass"])
        self.history["mean"].append(moments["mean"])
        self.history["variance"].append(moments["variance"])
        self.history["l2_norm"].append(l2)
        self.history["total_variation"].append(tv)
        self.history["min_value"].append(np.min(u))
        self.history["max_value"].append(np.max(u))
    
    def get_history(self) -> dict:
        """Get diagnostic history as arrays.
        
        Returns
        -------
        dict : Diagnostic history with numpy arrays
        """
        return {key: np.array(val) for key, val in self.history.items()}
    
    def clear(self):
        """Clear diagnostic history."""
        for key in self.history:
            self.history[key] = []
