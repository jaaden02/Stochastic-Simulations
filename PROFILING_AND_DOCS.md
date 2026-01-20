# Profiling and Documentation Guide

## Profiling Tools

### py-spy (Production Profiling)
Production-grade flame graph profiler for real-time performance analysis.

**Installation**: Included in `pyproject.toml` dev dependencies. Activate with `uv run`.

**Note**: On macOS, py-spy requires elevated permissions:
```bash
cd profiling
sudo uv run py-spy record -o profile.svg -- python profiling_examples.py
```

**Output**: SVG flamegraph saved to `profiling/profile.svg`

### scalene (CPU/GPU/Memory Profiling)
High-precision CPU and memory profiler with detailed per-line analysis.

**Installation**: Included in `pyproject.toml` dev dependencies. Activate with `uv run`.

```bash
cd profiling

# Run scalene profiling
uv run scalene run profiling_examples.py

# View results in terminal
uv run scalene view --cli
```

**Output**: JSON profile saved to `profiling/scalene-profile.json`

## API Documentation

Sphinx-generated API documentation is available in `docs_build/build/html/`.

### Building Documentation

```bash
cd docs_build
make html
```

### Viewing Documentation

Open `docs_build/build/html/index.html` in your browser to view:
- Complete API reference with all modules and classes
- Source code links for each documented item
- Search functionality for quick reference

### Documentation Modules

The API documentation includes:
1. **setup** - Grid, InitialCondition, and configuration classes
2. **boundary_conditions** - BoundaryConditions and related classes
3. **sde.solver** - PathSimulator and numerical integration schemes
4. **sde.kernel** - SDE mathematical kernels (Euler, Milstein, etc.)
5. **sde.selector** - Scheme selection utilities
6. **deterministic.solver** - DeterministicPDESolver class
7. **fokker_planck.solver** - FokkerPlanckSolver and SimulationEngine
8. **fokker_planck.selector** - FP scheme selection utilities
9. **results** - Result classes and comparison tools
10. **utilities** - Helper functions and utilities

## Profiling Examples

The `profiling/profiling_examples.py` script demonstrates profiling on:

### 1. Grid Creation
- 1D grids with 64 to 512 points
- Measures grid initialization performance

### 2. SDE Path Simulation
- 50 Brownian paths over 51 time steps
- Typical output shape: (50, 51, 1)
- ~150ms execution time

### 3. Fokker-Planck Simulation
- 256-point 1D grid with Gaussian initial condition
- Demonstrates FP solver setup and stability

### 4. 2D Simulation
- 64×64 2D grid creation
- 2D initial condition setup

## Performance Benchmarks

**Overall**: 344 tests in ~1.91 seconds (5.6ms average per test)

**By Category**:
- Unit tests: 124 tests in 0.85s (6.8ms avg)
- Integration tests: 66 tests in 1.41s (21.4ms avg)
- Numerical tests: 44 tests in 0.04s (0.9ms avg)
- Validation tests: 63 tests in 0.42s (6.7ms avg)
- Smoke tests: 47 tests in 0.03s (0.6ms avg)

**Slowest Operations**:
1. SDE comparison (0.46s)
2. Path simulator (0.42s)
3. Milstein scheme (0.20s)
4. 2D path simulation (0.17s)

## Quick Start

```bash
# Run profiling examples (no user input required)
cd profiling
uv run python profiling_examples.py

# Profile with scalene
uv run scalene run profiling_examples.py

# Profile with py-spy (macOS requires sudo)
sudo uv run py-spy record -o profile.svg -- python profiling_examples.py

# View scalene profile in terminal
uv run scalene view --cli

# Return to root
cd ..

# Generate API docs
cd docs_build && make html

# View all tests with coverage
uv run pytest tests/ --cov=src/stochlib --cov-report=html
```

## Development Tools Installed

- **py-spy 0.4.1** - Production profiling (profiles saved to `profiling/`)
- **scalene 2.0.1** - CPU/GPU/memory profiling (profiles saved to `profiling/`)
- **sphinx 9.1.0** - API documentation
- **sphinx-rtd-theme 3.1.0** - ReadTheDocs theme (fallback to alabaster)
- **27 supporting packages** - Documentation and profiling dependencies

## Project Structure

```
profiling/
├── profiling_examples.py      # Profiling example scripts
├── profile.svg                # py-spy flamegraph (generated)
└── scalene-profile.json       # scalene profile (generated)

docs_build/
├── source/                    # Sphinx source files
├── build/html/                # Generated HTML documentation
└── Makefile

src/stochlib/                  # Main source code
tests/                         # Test suite (344 tests)
examples/                      # Jupyter notebooks
```

## Notes

- All 344 tests pass successfully
- Code coverage is at 42%
- FP engine stability checks suppressed in profiling examples for headless execution
- Documentation builds without errors (19 warnings, mostly autodoc deprecations)
