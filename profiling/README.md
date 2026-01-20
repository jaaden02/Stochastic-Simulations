# Profiling Directory

This directory contains profiling examples and generated performance profiles for StochLib.

## Contents

- `profiling_examples.py` - Example scripts for profiling grid creation, SDE simulation, FP simulation, and 2D grids
- `profile.svg` - py-spy flamegraph (generated with `py-spy record`)
- `scalene-profile.json` - scalene CPU/memory profile (generated with `scalene run`)

## Quick Start

```bash
# Run all profiling examples (no user interaction)
uv run python profiling_examples.py

# Generate py-spy flamegraph (requires sudo on macOS)
sudo uv run py-spy record -o profile.svg -- python profiling_examples.py

# Generate scalene profile
uv run scalene run profiling_examples.py

# View scalene results in terminal
uv run scalene view --cli

# View in browser (scalene also generates HTML)
uv run scalene view
```

## Performance Insights

**From scalene profile:**
- **kernel.py (SDE solvers)**: 66.5% CPU - numerical integration bottleneck
- **distributions.py (FP plotting)**: 23.8% CPU - visualization overhead
- **Memory**: Peak 70 MB, linear growth, healthy scaling

**Key Findings:**
- SDE path simulation is CPU-bound (expected for numerical integration)
- Memory usage is not a constraint at current scales
- Total execution time: ~2 seconds for all examples

## See Also

- [PROFILING_AND_DOCS.md](../PROFILING_AND_DOCS.md) - Detailed profiling guide
- [README.md](../README.md) - Main project documentation
