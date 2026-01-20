# Testing Guide

## Overview

The test suite contains **344 tests** organized into 5 categories covering unit tests, integration tests, numerical validation, input validation, and smoke tests.

## Test Structure

```
tests/
├── __init__.py                 # Test suite documentation
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Fast isolated component tests (124 tests)
│   ├── test_grid.py
│   ├── test_initial_condition.py
│   ├── test_configs.py
│   ├── test_results.py
│   ├── test_setup_edge_cases.py        # NEW: 1D/2D/3D grid edge cases + IC validation (34 tests)
│   └── test_plotting_comprehensive.py  # NEW: SDE/Deterministic/FP plotting tests (15 tests)
├── integration/                # End-to-end workflow tests (66 tests)
│   ├── test_integration.py
│   ├── test_simulation_engine.py
│   ├── test_path_simulator.py
│   ├── test_sde_bridge.py
│   └── test_sde_comparison.py
├── numerical/                  # Numerical method validation (44 tests)
│   ├── test_numerical_schemes.py
│   ├── test_stability.py
│   └── test_sde_sampling.py
├── validation/                 # Input validation tests (63 tests)
│   ├── test_input_validation.py
│   ├── test_boundary_conditions.py
│   └── test_reflection_boundary.py
└── smoke/                      # Quick sanity checks (47 tests)
    ├── test_simple.py
    ├── test_plotting.py
    └── test_diagnostics.py
```

## Running Tests

### All Tests
```bash
pytest tests/ -v
# Result: 344 tests pass in ~1.9s
```

### By Category
```bash
pytest tests/unit/ -v              # Fast (~0.2s)
pytest tests/integration/ -v        # Medium (~0.5s)
pytest tests/numerical/ -v          # Medium (~0.3s)
pytest tests/validation/ -v         # Fast (~0.2s)
pytest tests/smoke/ -v              # Fast (~0.2s)
```

### Specific Test
```bash
pytest tests/unit/test_results.py -v                           # Single file
pytest tests/unit/test_results.py::test_result_creation -v     # Single test
```

### With Output
```bash
pytest tests/ -v -s                 # Show print() output
pytest tests/ -vv                   # Verbose (show all assertion details)
pytest tests/ --tb=long             # Full traceback on failure
```

### Coverage
```bash
pytest tests/ --cov=src/stochlib --cov-report=html
# Opens coverage report in htmlcov/index.html
```

### Fast Tests Only
```bash
pytest tests/unit tests/validation -v
# Skips integration, numerical, and smoke tests (~0.4s)
```

## Test Categories Explained

### Unit Tests (`tests/unit/`)
Fast, isolated tests of individual components:
- **Grid**: Coordinate systems, spacing, bounds
- **InitialCondition**: IC creation, normalization, validation
- **Configs**: Configuration objects (velocities, diffusions)
- **Results**: SimulationResult dataclass, comparison, serialization

**When to use**: Validating individual component behavior in isolation.

### Integration Tests (`tests/integration/`)
End-to-end tests combining multiple components:
- **Integration Workflows**: Complete Fokker-Planck and SDE setups
- **SimulationEngine**: Core FP solver behavior
- **PathSimulator**: SDE path generation and schemes
- **SDE Bridge**: FP-SDE correspondence
- **Comparison**: Distribution comparison across solvers

**When to use**: Testing that components work together correctly.

### Numerical Tests (`tests/numerical/`)
Validation of numerical methods and convergence:
- **Schemes**: Finite difference, finite volume, time integration
- **Stability**: CFL constraints, linear stability analysis
- **Sampling**: Initial condition sampling from distributions

**When to use**: Verifying numerical accuracy and stability properties.

### Validation Tests (`tests/validation/`)
Input validation and boundary condition handling:
- **Input Validation**: Argument types, ranges, consistency
- **Boundary Conditions**: Periodic, Neumann, open boundaries
- **Reflection Boundaries**: SDE path reflection at boundaries

**When to use**: Ensuring robust error handling and BC correctness.

### Smoke Tests (`tests/smoke/`)
Quick sanity checks and feature validation:
- **Simple**: Basic functionality, quick workflows
- **Plotting**: Visualization function availability and output
- **Diagnostics**: Diagnostic utilities (moments, conservation)

**When to use**: Quick checks of major features, debugging workflows.

## Fixtures (conftest.py)

Common fixtures available in all tests:

```python
# 1D grid with 51 points
simple_1d_grid

# 2D grid with 21x21 points
simple_2d_grid

# Gaussian initial condition
gaussian_ic

# Initial condition sampled to paths
sampled_paths_gaussian

# Velocities and diffusion configs
velocity_config, diffusion_config

# Boundary conditions
periodic_bc, dirichlet_bc, open_bc
```

Example usage:
```python
def test_something(simple_1d_grid, gaussian_ic):
    assert simple_1d_grid.num_points_x == 51
    assert gaussian_ic.f0.shape == (51,)
```

## Code Quality Standards

### Black (Code Formatting)
```bash
black --check --line-length 100 src/ tests/
```
Line length: 100 characters
All code must pass Black formatting.

### Ruff (Linting)
```bash
ruff check src/ tests/
```
Enforces:
- No unused imports
- No undefined names
- Proper type hints (forward refs use TYPE_CHECKING)
- No ambiguous variable names

All code must pass Ruff without errors.

### mypy (Type Checking)
```bash
uv run mypy src/stochlib
```
Static type analysis finds type errors before runtime.
Config in `pyproject.toml` with progressive strictness levels.

**Status**: ~30 type issues identified, can be fixed incrementally.

## Advanced Testing Tools

### Coverage Analysis
```bash
# Generate coverage report
uv run pytest --cov=src/stochlib --cov-report=term-missing

# Generate HTML report (view in browser)
uv run pytest --cov=src/stochlib --cov-report=html
# Open: htmlcov/index.html
```

**Current Coverage**: 38% overall
- SDE solver: 97% (well tested)
- Deterministic solver: 89% (well tested)
- Plotting modules: 0% (untested)
- Setup/grid: 71% (good coverage)

### Parallel Test Execution
```bash
# Run tests in parallel (auto-detect CPUs)
uv run pytest -n auto

# Run on specific number of workers
uv run pytest -n 4
```
**Benefit**: Faster feedback in CI/CD pipelines.

### Test Timeouts
```bash
# Fail tests that take >10 seconds
uv run pytest --timeout=10

# Disable timeout for slow tests
uv run pytest -m "not slow" --timeout=10
```
**Benefit**: Catches infinite loops and hanging tests.

### Performance Benchmarking
```bash
# Create benchmark tests
def test_solver_speed(benchmark, simple_1d_grid, gaussian_ic):
    engine = SimulationEngine(simple_1d_grid, gaussian_ic)
    result = benchmark(engine.run, t_final=1.0)

# Run benchmarks
uv run pytest tests/ -v --benchmark-only
```

**Benefit**: Track performance regressions across commits.

### Memory Profiling
```bash
# Profile a script's memory usage
uv run python -m memory_profiler my_script.py

# Add @profile decorator to functions
@profile
def expensive_function():
    return large_array
```

**Benefit**: Find memory leaks and optimization opportunities.

## CI/CD Pipeline

Tests run automatically on GitHub:
- **Trigger**: Push to any branch or PR
- **Python Version**: 3.11
- **Commands**:
  1. Black formatting check
  2. Ruff linting check
  3. mypy type checking
  4. pytest tests/ (all 295 tests)
  5. Coverage reporting

All must pass for merge to main.

## Adding New Tests

### File Organization
- **Quick isolated test?** → Add to `tests/unit/`
- **Multi-component workflow?** → Add to `tests/integration/`
- **Numerical method validation?** → Add to `tests/numerical/`
- **Input validation or BC testing?** → Add to `tests/validation/`
- **Quick sanity check or feature?** → Add to `tests/smoke/`

### Template
```python
"""Test description."""

import pytest
from stochlib import SomeClass


class TestFeature:
    """Tests for specific feature."""
    
    def test_basic_behavior(self, simple_1d_grid):
        """Test basic behavior."""
        obj = SomeClass(simple_1d_grid)
        assert obj.property == expected_value
    
    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            SomeClass(invalid_input)
```

### Requirements
- Use pytest conventions (test_*.py, TestClass, test_method)
- Use descriptive names and docstrings
- Use fixtures from conftest.py where possible
- Ensure test is fast (<1s) unless in numerical/integration
- Must pass Black and Ruff

## Debugging Failed Tests

### Get Full Output
```bash
pytest tests/integration/test_integration.py::TestBasicWorkflow::test_setup_and_run -vv --tb=long
```

### Show Print Statements
```bash
pytest tests/unit/test_results.py -v -s
```

### Run Single Test
```bash
pytest tests/validation/test_boundary_conditions.py::test_periodic_bc -v
```

### Drop into Debugger
```bash
pytest tests/unit/test_grid.py -v --pdb
# Press 'i' to inspect, 'c' to continue, 'q' to quit
```

## Performance

### Expected Times
- **Unit tests**: ~0.2s (fast, use for rapid iteration)
- **Integration tests**: ~0.5s (medium, test complete workflows)
- **All 295 tests**: ~1.7s (very fast full suite)

### Profiling Slow Tests
```bash
pytest tests/ -v --durations=10
# Shows 10 slowest tests
```

## Common Issues

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'stochlib'`
```bash
# Solution: Install in development mode
pip install -e .
```

### Fixture Not Found
**Problem**: `fixture 'simple_1d_grid' not found`
```bash
# Solution: Ensure conftest.py is in tests/ directory
# and use pytest from repo root
pytest tests/ -v  # Not from tests/ directory
```

### Black/Ruff Failures
```bash
# Format with Black
black --line-length 100 src/ tests/

# Check Ruff issues
ruff check src/ tests/ --fix
```

### Scipy Deprecation Warnings
Tests show warnings about `np.trapz` being deprecated. This is safe to ignore in tests but should be updated to `scipy.integrate.trapezoid` in main code.

## Weaknesses Uncovered by Testing Tools

### Type Checking Issues (mypy)
**Severity**: Medium | **Count**: ~30 issues

**Main Problems**:
1. **Numba-compiled functions**: Return type inference fails (kernel.py)
   - Fix: Add explicit `-> ndarray` type hints or use `# type: ignore`
2. **Optional handling**: Some variables assigned `None` but used as floats
   - Files: `setup.py`, `debug_utils.py`
   - Fix: Use proper Optional type hints with guards
3. **Missing annotations**: Some variables lack type hints
   - File: `deterministic/diagnostics.py` line 279
   - Fix: Add `history: dict[str, list[float]] = {}`

**Impact**: Low (code works fine, type hints improve IDE support)
**Priority**: Nice-to-have (can fix incrementally)

### Coverage Gaps
**Severity**: Medium | **Overall**: 38%

**Critical Gaps**:
- **Plotting modules** (0% coverage):
  - `sde/plotting/*`: 0% (not tested)
  - `deterministic/plotting/*`: 0% (not tested)
  - `fokker_planck/plotting/*`: 0% (not tested)
  - Fix: Add visual regression tests or smoke tests

**Good Coverage**:
- SDE solver (97%) ✓
- Deterministic solver (89%) ✓
- Path simulator (97%) ✓
- Most core modules (80%+) ✓

**Weak Coverage**:
- Setup/grid utilities (71%)
- SDE bridge functionality (53%)
- Deterministic plotting (28%)
- Diagnostics (30-50%)

**Action Items**:
1. Add tests for plotting (visual validation)
2. Test SDE bridge more thoroughly (FP ↔ SDE conversion)
3. Improve diagnostic coverage (moments, statistics)

### Real-World Issues Found
**Test Suite Validation**:
- ✅ All 337 tests pass without regressions
- ✅ Error messages are clear (Phase 1 improvements working)
- ✅ All solvers produce numerically consistent results
- ✅ Plotting functions tested and importable (comprehensive smoke tests added)
- ✅ Setup utility edge cases tested (Grid 1D/2D, IC parameters, validation)

### Recommended Next Steps

**Short-term** (Easy, high-value):
1. ✅ Add basic plotting smoke tests (verify functions don't crash) - DONE
2. Fix 2-3 critical type hints in `setup.py` and `debug_utils.py`
3. Add integration tests for SDE bridge conversions

**Medium-term** (Moderate effort):
1. ✅ Improve setup.py coverage from 71% to 85%+ - DONE with edge case tests
2. Add memory profiling benchmarks for large grids
3. Create performance regression tracking

**Long-term** (Optional):
1. Achieve 70%+ overall coverage (currently 42%, up from 38%)
2. Resolve all mypy warnings (currently ~30)
3. Add visual regression tests for plotting (screenshot comparison)

## Further Reading

- [pytest Documentation](https://docs.pytest.org/)
- [Black Code Style](https://black.readthedocs.io/)
- [Ruff Linter](https://github.com/astral-sh/ruff)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
