import logging
import os
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(level: Optional[int] = None, verbose: bool = False) -> logging.Logger:
    """Configure a shared stochlib logger with a single stream handler.

    - If ``level`` is provided, it takes precedence.
    - Else if ``verbose`` is True, uses DEBUG; otherwise INFO.
    - An environment variable ``STOCHLIB_LOGLEVEL`` (e.g., "DEBUG") can override defaults.
    """
    env_level = os.getenv("STOCHLIB_LOGLEVEL")
    resolved_level = level
    if resolved_level is None:
        if env_level:
            resolved_level = getattr(logging, env_level.upper(), logging.INFO)
        else:
            resolved_level = logging.DEBUG if verbose else logging.INFO

    logger = logging.getLogger("stochlib")
    logger.setLevel(resolved_level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        handler.setLevel(resolved_level)
        logger.addHandler(handler)

    logger.debug("Logging configured at level %s", logging.getLevelName(resolved_level))
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    base = "stochlib"
    full_name = f"{base}.{name}" if name else base
    return logging.getLogger(full_name)
