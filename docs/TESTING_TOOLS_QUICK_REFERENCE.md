# Quick Command Reference

## Running Tests with New Tools

### Basic Test Commands
```bash
# All tests
uv run pytest

# Tests in parallel (faster)
uv run pytest -n auto

# With timeout protection
uv run pytest --timeout=10
```

### Coverage & Quality
```bash
# Coverage report (terminal)
uv run pytest --cov=src/stochlib --cov-report=term-missing

# Coverage report (HTML browser)
uv run pytest --cov=src/stochlib --cov-report=html
# Then open: htmlcov/index.html

# Type checking
uv run mypy src/stochlib

# Performance profiling
uv run pytest tests/ -v --durations=10
```

### Memory Analysis
```bash
# Profile a script
uv run python -m memory_profiler my_script.py

# Profile a test
@profile
def test_memory_intensive():
    big_array = np.zeros((10000, 10000))
    # ...
```

## Current Status

| Tool | Status | Key Findings |
|------|--------|--------------|
| mypy | ✅ Installed | ~30 type hints issues (low priority) |
| pytest-cov | ✅ Installed | 38% coverage, plotting at 0% |
| pytest-xdist | ✅ Installed | Parallel execution 3.27s (very fast) |
| pytest-timeout | ✅ Installed | No hanging tests detected |
| pytest-benchmark | ✅ Installed | Ready for perf regressions |
| memory-profiler | ✅ Installed | Ready for memory analysis |

## Weaknesses Found

### Type Issues (mypy)
- Numba functions need explicit return types
- Some Optional variables handled unsafely
- 2-3 critical issues in setup.py

### Coverage Gaps
- Plotting modules: 0% (untested)
- SDE bridge: 53% (needs improvement)
- Diagnostics: 30-50% (basic testing only)

### Recommended Fixes
1. **High**: Add plotting smoke tests
2. **Medium**: Fix type hints in setup.py
3. **Low**: Improve coverage to 70%+
