"""Developer debugging utilities for stochlib.

Provides debug mode support, assertion helpers, state dumping, and diagnostic tools
for developers troubleshooting simulation issues.

Environment Variables:
    STOCHLIB_DEBUG: Enable debug mode (1, true, yes). Enables extra checks and diagnostics.
    STOCHLIB_DUMP_DIR: Directory for state dumps (default: ./debug_dumps/)
"""

import os
import json
from functools import wraps
import numpy as np
from typing import Any, Optional, Dict

from .logging_utils import get_logger

logger = get_logger("debug")

# Debug mode control
DEBUG_MODE = os.getenv("STOCHLIB_DEBUG", "").lower() in ("1", "true", "yes")
DUMP_DIR = os.getenv("STOCHLIB_DUMP_DIR", "./debug_dumps/")


def enable_debug() -> None:
    """Enable debug mode (can be called before imports)."""
    global DEBUG_MODE
    DEBUG_MODE = True
    logger.info("Debug mode ENABLED. Extra checks and diagnostics active.")


def disable_debug() -> None:
    """Disable debug mode."""
    global DEBUG_MODE
    DEBUG_MODE = False
    logger.info("Debug mode DISABLED.")


def is_debug_mode() -> bool:
    """Check if debug mode is active."""
    return DEBUG_MODE


def require_debug_mode(func):
    """Decorator: Only run function in debug mode.

    In non-debug mode, logs message and returns None.

    Example:
        @require_debug_mode
        def check_solution_bounds(u):
            assert np.all(u >= 0), "Negative values detected"
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not DEBUG_MODE:
            logger.debug(f"Skipped {func.__name__} (requires STOCHLIB_DEBUG=1)")
            return None
        return func(*args, **kwargs)

    return wrapper


# ============================================================================
# ASSERTION HELPERS
# ============================================================================


@require_debug_mode
def assert_finite(arr: np.ndarray, name: str = "array", raise_error: bool = True) -> bool:
    """Assert array contains only finite values (no NaN or Inf).

    Parameters
    ----------
    arr : ndarray
        Array to check
    name : str
        Name for error messages
    raise_error : bool
        If True, raise ValueError on non-finite values

    Returns
    -------
    bool
        True if all finite, False otherwise

    Raises
    ------
    ValueError
        If raise_error=True and non-finite values found

    Example:
        >>> u = np.array([1, 2, np.nan, 4])
        >>> assert_finite(u, "solution")  # Raises ValueError
    """
    if not np.all(np.isfinite(arr)):
        non_finite_mask = ~np.isfinite(arr)
        non_finite_vals = arr[non_finite_mask]
        msg = (
            f"{name} contains {np.sum(non_finite_mask)} non-finite values "
            f"({np.sum(np.isnan(non_finite_vals))} NaN, "
            f"{np.sum(np.isinf(non_finite_vals))} Inf)"
        )
        logger.error(msg)
        if raise_error:
            raise ValueError(msg)
        return False
    return True


@require_debug_mode
def assert_positive(arr: np.ndarray, name: str = "array", raise_error: bool = True) -> bool:
    """Assert array contains only positive values.

    Useful for probability distributions, positive definite matrices, etc.

    Parameters
    ----------
    arr : ndarray
        Array to check
    name : str
        Name for error messages
    raise_error : bool
        If True, raise ValueError on negative/zero values

    Returns
    -------
    bool
        True if all positive, False otherwise
    """
    if np.any(arr <= 0):
        negative_mask = arr <= 0
        msg = (
            f"{name} contains {np.sum(negative_mask)} non-positive values "
            f"(min={np.min(arr)}, max={np.max(arr)})"
        )
        logger.error(msg)
        if raise_error:
            raise ValueError(msg)
        return False
    return True


@require_debug_mode
def assert_shape(arr: np.ndarray, expected_shape: tuple, name: str = "array") -> bool:
    """Assert array has expected shape.

    Parameters
    ----------
    arr : ndarray
        Array to check
    expected_shape : tuple
        Expected shape
    name : str
        Name for error messages

    Returns
    -------
    bool
        True if shape matches

    Raises
    ------
    ValueError
        If shape doesn't match
    """
    if arr.shape != expected_shape:
        msg = f"{name} shape {arr.shape} != expected {expected_shape}"
        logger.error(msg)
        raise ValueError(msg)
    return True


# ============================================================================
# STATE DUMPING & INSPECTION
# ============================================================================


def dump_state(obj: Any, filepath: Optional[str] = None, name: Optional[str] = None) -> str:
    """Dump object state to JSON file for inspection.

    Useful for debugging: capture the state of objects at failure point.

    Parameters
    ----------
    obj : Any
        Object to dump (must be JSON-serializable or have __dict__)
    filepath : str, optional
        Output file path. If None, uses STOCHLIB_DUMP_DIR/{name}.json
    name : str, optional
        Object name for filename (used if filepath is None)

    Returns
    -------
    str
        Path to dump file

    Example:
        >>> engine = SimulationEngine(...)
        >>> try:
        ...     result = engine.run(...)
        ... except Exception as e:
        ...     dump_state(engine, name="failed_engine")
        ...     raise
    """
    # Determine output path
    if filepath is None:
        os.makedirs(DUMP_DIR, exist_ok=True)
        obj_name = name or obj.__class__.__name__
        filepath = os.path.join(DUMP_DIR, f"{obj_name}.json")

    # Serialize object
    state = {"type": obj.__class__.__name__, "module": obj.__class__.__module__, "attrs": {}}

    for key, val in obj.__dict__.items():
        try:
            # Try JSON serialization
            json.dumps(val)
            state["attrs"][key] = val
        except (TypeError, ValueError):
            # Fall back to string representation
            if isinstance(val, np.ndarray):
                state["attrs"][key] = {
                    "_type": "ndarray",
                    "shape": val.shape,
                    "dtype": str(val.dtype),
                    "min": float(np.min(val)) if val.size > 0 else None,
                    "max": float(np.max(val)) if val.size > 0 else None,
                    "mean": float(np.mean(val)) if val.size > 0 else None,
                }
            else:
                state["attrs"][key] = str(val)

    # Write to file
    with open(filepath, "w") as f:
        json.dump(state, f, indent=2)

    logger.info(f"Dumped {obj.__class__.__name__} state to {filepath}")
    return filepath


def inspect_array(arr: np.ndarray, name: str = "array") -> Dict[str, Any]:
    """Inspect array properties for debugging.

    Parameters
    ----------
    arr : ndarray
        Array to inspect
    name : str
        Name for logging

    Returns
    -------
    dict
        Inspection results with shape, dtype, statistics, etc.
    """
    result = {
        "name": name,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "size": arr.size,
        "nbytes": arr.nbytes,
    }

    if arr.size > 0:
        result.update(
            {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "has_nan": bool(np.any(np.isnan(arr))),
                "has_inf": bool(np.any(np.isinf(arr))),
                "negative_count": int(np.sum(arr < 0)),
            }
        )

    return result


def print_array_info(*arrays, **named_arrays) -> None:
    """Print detailed info about arrays for debugging.

    Example:
        >>> u = np.array([1, 2, 3])
        >>> v = np.array([4, 5, 6])
        >>> print_array_info(u, solution=v)
        # Prints detailed info about both arrays
    """
    all_arrays = [(f"arg{i}", arr) for i, arr in enumerate(arrays)]
    all_arrays.extend(named_arrays.items())

    for name, arr in all_arrays:
        if isinstance(arr, np.ndarray):
            info = inspect_array(arr, name)
            logger.info(f"Array '{name}': {info}")
        else:
            logger.warning(f"'{name}' is not an ndarray: {type(arr)}")


# ============================================================================
# PERFORMANCE PROFILING
# ============================================================================


@require_debug_mode
def profile_function(func):
    """Decorator: Profile function execution (requires debug mode).

    Logs execution time and memory usage.

    Example:
        @profile_function
        def expensive_computation(n):
            return np.sum(np.arange(n))
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        import tracemalloc

        tracemalloc.start()
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            logger.debug(
                f"{func.__name__}: time={elapsed:.4f}s, " f"memory={peak / 1e6:.1f} MB peak"
            )
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            tracemalloc.stop()
            logger.error(f"{func.__name__} failed after {elapsed:.4f}s: {e}")
            raise

    return wrapper


# ============================================================================
# BREAKPOINT HELPERS
# ============================================================================


def breakpoint_on_condition(condition: bool, message: str = "") -> None:
    """Drop into debugger if condition is True (debug mode only).

    Parameters
    ----------
    condition : bool
        If True, triggers breakpoint
    message : str
        Message to log before breaking

    Example:
        >>> x = some_computation()
        >>> breakpoint_on_condition(x > threshold, f"Value {x} exceeds {threshold}")
    """
    if DEBUG_MODE and condition:
        logger.warning(f"BREAKPOINT: {message}")
        breakpoint()


def breakpoint_if_nan(arr: np.ndarray, name: str = "array") -> None:
    """Drop into debugger if array contains NaN (debug mode only)."""
    if DEBUG_MODE and np.any(np.isnan(arr)):
        logger.warning(f"BREAKPOINT: NaN detected in {name}")
        print_array_info(**{name: arr})
        breakpoint()


def breakpoint_if_inf(arr: np.ndarray, name: str = "array") -> None:
    """Drop into debugger if array contains Inf (debug mode only)."""
    if DEBUG_MODE and np.any(np.isinf(arr)):
        logger.warning(f"BREAKPOINT: Inf detected in {name}")
        print_array_info(**{name: arr})
        breakpoint()


# ============================================================================
# CONTEXT MANAGERS
# ============================================================================


class debug_context:
    """Context manager for debug-specific code blocks.

    Example:
        with debug_context("expensive check"):
            validate_solution(u)  # Only runs in debug mode
    """

    def __init__(self, label: str):
        self.label = label
        self.start_time: Optional[float] = None

    def __enter__(self):
        if DEBUG_MODE:
            import time

            self.start_time = time.time()
            logger.debug(f"DEBUG: Entering {self.label}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if DEBUG_MODE:
            import time

            elapsed = time.time() - (self.start_time or 0.0)
            if exc_type is not None:
                logger.error(f"DEBUG: {self.label} failed after {elapsed:.4f}s: {exc_val}")
            else:
                logger.debug(f"DEBUG: Exited {self.label} ({elapsed:.4f}s)")
        return False


__all__ = [
    "DEBUG_MODE",
    "enable_debug",
    "disable_debug",
    "is_debug_mode",
    "require_debug_mode",
    "assert_finite",
    "assert_positive",
    "assert_shape",
    "dump_state",
    "inspect_array",
    "print_array_info",
    "profile_function",
    "breakpoint_on_condition",
    "breakpoint_if_nan",
    "breakpoint_if_inf",
    "debug_context",
]
