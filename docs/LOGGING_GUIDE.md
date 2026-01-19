# Logging Usage Guide

Quick reference for using stochlib's logging system.

## Quick Start

```python
from stochlib import configure_logging, get_logger
import logging

# Initialize logging at program start
configure_logging(level=logging.INFO, verbose=True)

# Get a logger for your module
logger = get_logger("my_simulation")

# Use it
logger.info("Starting PDE solver")
logger.debug("Grid points: %d", 1000)
logger.warning("CFL number exceeded threshold")
logger.error("Solution diverged")
```

## Logging Levels

- **DEBUG** (10): Detailed diagnostics, variable values, function calls
- **INFO** (20): General informational messages, progress updates
- **WARNING** (30): Warning messages, potential issues
- **ERROR** (40): Error messages, failures
- **CRITICAL** (50): Critical failures, program cannot continue

## Configuration

### By Code

```python
from stochlib import configure_logging
import logging

# Minimal setup (INFO level, no timestamps)
configure_logging()

# Debug mode with timestamps
configure_logging(level=logging.DEBUG, verbose=True)

# With file logging
configure_logging(
    level=logging.INFO,
    verbose=True,
    log_file="simulation.log"
)

# Custom level
configure_logging(level=logging.WARNING)
```

### By Environment Variables

```bash
# Set logging level
export STOCHLIB_LOGLEVEL=DEBUG

# Set log file
export STOCHLIB_LOG_FILE=/var/log/stochlib.log

# Both together
export STOCHLIB_LOGLEVEL=INFO
export STOCHLIB_LOG_FILE=~/logs/sim.log
python my_script.py
```

Environment variables override function arguments!

## Getting Loggers

```python
from stochlib import get_logger

# Root logger (no name)
logger = get_logger()

# Named loggers (recommended for modules)
logger = get_logger("my_module")
logger = get_logger("simulation.pde_solver")
logger = get_logger("analysis.plotting")
```

Logger names should use dot notation to create hierarchies:
- `solver` - Top-level module
- `solver.fokker_planck` - Submodule
- `solver.fokker_planck.kernel` - Sub-submodule

## Format Examples

### Default Format
```
[INFO    ] stochlib.my_module: Starting simulation
[DEBUG   ] stochlib.my_module: Grid created with 1000 points
[WARNING ] stochlib.my_module: CFL number = 1.5 (exceeds 1.0)
[ERROR   ] stochlib.my_module: Solver diverged after 50 steps
```

### Verbose Format (with timestamps)
```
2026-01-19 14:23:45,123 [INFO    ] stochlib.my_module: Starting simulation
2026-01-19 14:23:45,145 [DEBUG   ] stochlib.my_module: Grid created with 1000 points
2026-01-19 14:23:46,234 [WARNING ] stochlib.my_module: CFL number = 1.5
```

### Debug Format (with function info)
```
2026-01-19 14:23:45,123 [DEBUG   ] stochlib.my_module:setup:42: Initializing Grid
2026-01-19 14:23:45,145 [DEBUG   ] stochlib.my_module:step:156: Computing flux
```

## Performance Tracking

Use the `@log_performance` decorator to automatically track execution time:

```python
from stochlib import get_logger, log_performance

logger = get_logger("my_module")

@log_performance(logger)
def solve_pde(grid, ic, t_end):
    # Code here...
    return solution

# Output:
# [DEBUG] stochlib.my_module: Calling solve_pde with args=(...), kwargs={}
# [DEBUG] stochlib.my_module: solve_pde completed in 2.3456s
```

If an exception occurs:
```python
# [ERROR] stochlib.my_module: solve_pde failed after 0.1234s: RuntimeError...
```

## Common Patterns

### Module Template
```python
"""Module description."""
from ..logging_utils import get_logger

logger = get_logger("my_module")

class MySolver:
    def __init__(self, grid, ic):
        logger.debug("Initializing MySolver")
        self.grid = grid
        self.ic = ic
    
    def step(self, dt):
        logger.debug(f"Taking step with dt={dt}")
        # ... computation ...
        logger.info(f"Step completed, new time t={self.t}")
        return solution
```

### Function Logging
```python
def solve_problem(config):
    logger.info(f"Solving problem with config: {config}")
    
    try:
        result = expensive_computation(config)
        logger.info(f"Solution found: {result}")
        return result
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        raise
    finally:
        logger.debug("Cleaning up resources")
```

### Diagnostic Output
```python
def validate_solution(u, u_exact):
    error = compute_error(u, u_exact)
    logger.info(f"L2 error: {error:.2e}")
    
    if error > tolerance:
        logger.warning(f"Error {error} exceeds tolerance {tolerance}")
    else:
        logger.debug(f"Solution validated successfully")
    
    return error < tolerance
```

## Suppressing Logs

To suppress logs from a specific module:

```python
import logging

# Quiet a noisy module
logging.getLogger("stochlib.verbose_module").setLevel(logging.WARNING)

# Quiet all stochlib except errors
logging.getLogger("stochlib").setLevel(logging.ERROR)
```

## File Logging

Logs to both console and file:

```python
from stochlib import configure_logging
import logging

configure_logging(
    level=logging.INFO,
    log_file="results/simulation.log"
)

# Console gets: [INFO    ] stochlib.module: message
# File gets:    2026-01-19 14:23:45,123 [INFO    ] stochlib.module: message
```

## Structured Information

Include context in log messages:

```python
logger.info(
    "Simulation progress",
    extra={
        "time": 1.5,
        "step": 100,
        "grid_points": 1000,
        "error": 1.2e-4,
    }
)

# For simple key=value output:
logger.info(f"t={1.5} step={100} error={1.2e-4}")
```

## Best Practices

1. **Use appropriate levels**: Debug for internal state, Info for progress
2. **Include context**: What input, what is being computed, what went wrong
3. **Use string formatting**: `logger.info(f"value={x}")` not string concatenation
4. **Name loggers consistently**: Use module-based hierarchical names
5. **Log at function entry/exit**: Great for debugging complex flows
6. **Log exceptional cases**: Warnings and errors should stand out
7. **Don't log secrets**: Never log passwords, tokens, private data

## Troubleshooting

**No log output?**
```python
# Ensure configure_logging was called
from stochlib import configure_logging
configure_logging()  # Call before other stochlib code
```

**Duplicate logs?**
```python
# Don't add handlers manually, use configure_logging
# Don't set logger propagate=True for stochlib loggers
```

**Too much output?**
```python
# Reduce verbosity
import logging
from stochlib import configure_logging

configure_logging(level=logging.WARNING)
```

**Missing file logs?**
```python
# Ensure log file path is writable
import os
log_dir = "results"
os.makedirs(log_dir, exist_ok=True)

from stochlib import configure_logging
configure_logging(log_file=f"{log_dir}/sim.log")
```
