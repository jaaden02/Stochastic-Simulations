---
name: Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: bug
assignees: ''
---

## Description
A clear and concise description of the bug.

## To Reproduce
Steps to reproduce the behavior:
```python
# Minimal code example that reproduces the bug
from stochlib.setup import Grid
grid = Grid(x_start=0.0, x_end=1.0, num_points_x=101)
# ... rest of code
```

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened. Include full error traceback:
```
Traceback (most recent call last):
  ...
```

## Environment
- OS: [e.g., macOS 14.0, Ubuntu 22.04, Windows 11]
- Python version: [e.g., 3.12.1]
- StochLib version: [e.g., 0.1.0]
- Installation method: [pip, conda, source]

**Versions of dependencies:**
```bash
pip list | grep -E "numpy|numba|scipy"
```

## Additional Context
Any other context about the problem (screenshots, related issues, etc.)

## Possible Solution
(Optional) Suggest a fix or workaround if you have one.
