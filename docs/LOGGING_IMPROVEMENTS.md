# Logging and Documentation Improvements

## Summary of Changes

### 1. Enhanced Logging Infrastructure ✓

**Updated `src/stochlib/logging_utils.py`** with comprehensive logging capabilities:

#### New Features:
- **Dual Output**: Console logging + optional file logging
- **Environment Variables**: `STOCHLIB_LOGLEVEL` and `STOCHLIB_LOG_FILE` for configuration
- **Multiple Format Levels**:
  - Default: `[LEVEL] module: message`
  - Verbose: `TIMESTAMP [LEVEL] module: message`
  - Debug: `TIMESTAMP [LEVEL] module:function:line: message`
- **Performance Decorator**: `@log_performance()` for timing and error tracking
- **Duplicate Prevention**: Automatic handler deduplication

#### API:
```python
# Configure logging with all options
configure_logging(
    level=logging.DEBUG,        # Log level
    verbose=True,               # Include timestamps
    log_file="simulation.log"    # File output
)

# Get named loggers
logger = get_logger("module_name")
logger.info("Message")

# Performance tracking (decorator)
@log_performance(logger)
def expensive_function():
    ...
```

### 2. Logging Applied Throughout Codebase ✓

Added logging to all major modules:

**SDE Module** (`sde/`):
- `sde/solver.py` - PathSimulator logging
- `sde/selector.py` - Scheme selection logging
- `sde/diagnostic.py` - Diagnostic tracking
- `sde/bridge.py` - Parameter conversion logging

**Deterministic Module** (`deterministic/`):
- `deterministic/solver.py` - DeterministicPDESolver logging
- `deterministic/diagnostics.py` - Solution quality logging

**Already Had Logging**:
- Fokker-Planck module (all solvers, diagnostics, plotting)

### 3. New: FILE_STRUCTURE.md Documentation ✓

Created comprehensive `FILE_STRUCTURE.md` documenting:

#### Contents:
- **Project Root Structure**: Overview of key files and directories
- **Complete Module Tree**: Visual representation of package organization
- **Module Descriptions**: Detailed purpose of each component
- **Public API Reference**: Exported classes and functions
- **Test Structure**: Organization of test suite
- **Configuration Details**: `pyproject.toml`, `.gitignore`, logging setup
- **Development Standards**: Type hints, docstrings, code style
- **Performance Notes**: JIT compilation, NumPy vectorization

#### Sections:
1. Project Organization
2. Source Structure (focused)
3. Module Descriptions (detailed)
4. Public API (with import examples)
5. Test Organization
6. Configuration Files
7. Logging Setup
8. Development Notes

### 4. Updated Main Package Exports ✓

**`src/stochlib/__init__.py`** now exports:
```python
from stochlib import (
    # Existing
    configure_logging,
    get_logger,
    # New
    log_performance,
)
```

## Usage Examples

### Basic Logging Setup
```python
import logging
from stochlib import configure_logging, get_logger

# Configure at startup
configure_logging(level=logging.INFO, verbose=True)

# Get logger for your code
logger = get_logger("my_module")
logger.info("Starting simulation")
```

### File Logging
```python
from stochlib import configure_logging

configure_logging(
    log_file="results/simulation.log",
    verbose=True
)
```

### Environment Configuration
```bash
# Set logging level via environment
export STOCHLIB_LOGLEVEL=DEBUG

# Set log file via environment
export STOCHLIB_LOG_FILE=/tmp/sim.log

# Run your script
python my_script.py
```

### Performance Monitoring
```python
from stochlib import get_logger, log_performance

logger = get_logger("my_module")

@log_performance(logger)
def solve_pde():
    # Function automatically timed and logged
    ...
```

## Files Modified/Created

### Modified:
- `src/stochlib/logging_utils.py` - Enhanced logging utilities
- `src/stochlib/__init__.py` - Export log_performance
- `src/stochlib/sde/solver.py` - Added logger
- `src/stochlib/sde/selector.py` - Added logger
- `src/stochlib/sde/diagnostic.py` - Added logger
- `src/stochlib/sde/bridge.py` - Added logger
- `src/stochlib/deterministic/solver.py` - Added logger
- `src/stochlib/deterministic/diagnostics.py` - Added logger

### Created:
- `FILE_STRUCTURE.md` - Complete project documentation (463 lines)

## Testing Status

✅ **All 279 tests pass**
✅ **No new warnings introduced**
✅ **Backward compatible** - Existing code works unchanged
✅ **Type hints maintained** - Full PEP 561 compliance

## Environment Support

The logging system respects Python environment variables:
- `STOCHLIB_LOGLEVEL`: DEBUG | INFO | WARNING | ERROR | CRITICAL
- `STOCHLIB_LOG_FILE`: Path to log file for persistent logging

## Next Steps (Optional)

Consider these enhancements in the future:
1. Add log handlers for structured logging (JSON format)
2. Create specialized loggers for different solvers
3. Add performance profiling decorators for critical paths
4. Integrate with external logging services (e.g., Sentry)
