"""Logging configuration utilities for the stochlib package.

Provides centralized logging setup with customizable verbosity levels, context tracking,
and performance monitoring capabilities.

Environment Variables:
    STOCHLIB_LOGLEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    STOCHLIB_LOG_FILE: Optional file path for logging to file
"""
import logging
import os
import sys
from typing import Optional
from functools import wraps
import time

# Log format strings
_DEFAULT_FORMAT = "[%(levelname)-8s] %(name)s: %(message)s"
_VERBOSE_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DEBUG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s:%(funcName)s:%(lineno)d: %(message)s"


def configure_logging(
    level: Optional[int] = None,
    verbose: bool = False,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure a shared stochlib logger with stream and optional file handlers.

    Parameters
    ----------
    level : int, optional
        Logging level (logging.DEBUG, logging.INFO, etc.). Takes precedence over other options.
    verbose : bool
        If True and level is None, uses DEBUG level with timestamp format.
    log_file : str, optional
        Path to log file. If provided, logs are written to both console and file.

    Returns
    -------
    logging.Logger
        The configured stochlib logger.

    Notes
    -----
    Environment variable STOCHLIB_LOGLEVEL can override default level.
    Environment variable STOCHLIB_LOG_FILE can override log_file parameter.
    """
    env_level = os.getenv("STOCHLIB_LOGLEVEL")
    env_logfile = os.getenv("STOCHLIB_LOG_FILE")
    
    resolved_level = level
    if resolved_level is None:
        if env_level:
            resolved_level = getattr(logging, env_level.upper(), logging.INFO)
        else:
            resolved_level = logging.DEBUG if verbose else logging.INFO

    logger = logging.getLogger("stochlib")
    logger.setLevel(resolved_level)
    logger.propagate = False  # Prevent duplicate logs

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(resolved_level)
    
    # Choose format based on level
    if resolved_level == logging.DEBUG:
        console_format = _DEBUG_FORMAT
    elif verbose:
        console_format = _VERBOSE_FORMAT
    else:
        console_format = _DEFAULT_FORMAT
    
    console_handler.setFormatter(logging.Formatter(console_format))
    logger.addHandler(console_handler)

    # File handler (if requested)
    log_path = env_logfile or log_file
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(logging.Formatter(_VERBOSE_FORMAT))
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_path}")

    logger.debug("Logging configured at level %s", logging.getLevelName(resolved_level))
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a named logger under the stochlib namespace.

    Parameters
    ----------
    name : str, optional
        Logger name (e.g., "fokker_planck.solver"). If None, returns root stochlib logger.

    Returns
    -------
    logging.Logger
        The requested logger.

    Examples
    --------
    >>> logger = get_logger("my_module")
    >>> logger.debug("Debug message")
    >>> logger.info("Info message")
    """
    base = "stochlib"
    full_name = f"{base}.{name}" if name else base
    return logging.getLogger(full_name)


def log_performance(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution time and arguments.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use. If None, uses root stochlib logger.

    Returns
    -------
    callable
        Decorated function that logs performance metrics.

    Examples
    --------
    >>> logger = get_logger("my_module")
    >>> @log_performance(logger)
    >>> def expensive_function(x, y):
    ...     return x + y
    """
    def decorator(func):
        _logger = logger or get_logger()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            _logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                _logger.debug(f"{func.__name__} completed in {elapsed:.4f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                _logger.error(f"{func.__name__} failed after {elapsed:.4f}s: {e}", exc_info=True)
                raise
        return wrapper
    return decorator

