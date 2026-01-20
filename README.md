# StochLib: Stochastic Simulations Library

A Python library for modeling stochastic processes through **Fokker-Planck PDEs** and **Monte Carlo SDE path simulation**.

[![Tests](https://github.com/jasperaden/Stochastic-Simulations/workflows/Tests/badge.svg)](https://github.com/jasperaden/Stochastic-Simulations/actions)
[![codecov](https://codecov.io/gh/jasperaden/Stochastic-Simulations/branch/main/graph/badge.svg)](https://codecov.io/gh/jasperaden/Stochastic-Simulations)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Features

- **Fokker-Planck PDE Solver**: High-performance numerical solution of forward Fokker-Planck equations for probability evolution
- **SDE Path Simulator**: Monte Carlo simulation with automatic scheme selection (Euler-Maruyama / Milstein)
- **Deterministic PDE Solver**: Pure advection equation solver for comparison
- **Comprehensive Boundary Conditions**: Periodic, no-flux (Neumann), and open (outflow) boundary types
- **Automatic Stability Analysis**: CFL constraint checking and adaptive time-stepping
- **Production Logging**: File output, environment variable control, and performance decorators

## Quick Start

### Installation

```bash
git clone <repository-url>
cd Stochastic-Simulations
pip install -e .
```

### Fokker-Planck Example

```python
from stochlib.setup import Grid, InitialCondition, VelocitiesConfig, DiffusionConfig
from stochlib.boundary_conditions import BoundaryConditions
from stochlib.fokker_planck import SimulationEngine
import numpy as np

# 1D diffusion setup
grid = Grid(x_start=0.0, x_end=1.0, num_points_x=201)
ic = InitialCondition(grid, func_type="gaussian", x0=0.5, sigma_x=0.05)
velocities = VelocitiesConfig(grid, mu_x=0.0)  # No drift
diffusions = DiffusionConfig(grid, axes=['x'], constants={'x': 0.1})
bc = BoundaryConditions(grid, bc_x='periodic')

# Run simulation
engine = SimulationEngine(grid, velocities, diffusions, bc)
t_array = np.linspace(0, 0.1, 101)
result = engine.run(f0=ic.f0, t_array=t_array, confirm_run=False)

print(f"Final distribution shape: {result['final'].shape}")
print(f"Snapshots saved: {len(result['snapshots'])}")
```

### SDE Path Simulation Example

```python
from stochlib.sde import PathSimulator
import numpy as np

# Define drift and diffusion
def drift(x, t):
    return 0.5 * x  # Linear drift

def diffusion(x, t):
    return 0.3 if not isinstance(x, np.ndarray) else np.full_like(x, 0.3)

# Simulate 1000 paths
sim = PathSimulator(drift=drift, diffusion=diffusion, scheme='auto')
x0 = np.zeros((1000, 1))
t_array = np.linspace(0, 1.0, 101)
result = sim.simulate(x0=x0, t_array=t_array, save_paths=True)

print(f"Final mean: {result['mean'][-1, 0]:.4f}")
print(f"Final variance: {result['var'][-1, 0]:.4f}")
```

## Project Structure

```
stochlib/
├── src/stochlib/
│   ├── setup.py                 # Grid, InitialCondition, configs
│   ├── boundary_conditions.py   # BC implementations
│   ├── logging_utils.py         # Enhanced logging
│   ├── fokker_planck/           # FP PDE solver
│   │   ├── solver.py            # FokkerPlanckSolver
│   │   ├── selector.py          # SimulationEngine, StabilityAnalyzer
│   │   ├── kernel.py            # JIT-compiled stepping
│   │   └── plotting/            # Visualization tools
│   ├── sde/                     # SDE path simulator
│   │   ├── solver.py            # PathSimulator
│   │   ├── kernel.py            # Euler-Maruyama, Milstein
│   │   ├── selector.py          # StepSchemeAdvisor
│   │   └── plotting/            # Trajectory plots
│   └── deterministic/           # Deterministic PDE solver
├── tests/                       # 279 comprehensive tests
├── examples/                    # 7 Jupyter notebooks
└── docs/                        # Additional documentation
```

## Examples & Tutorials

We provide 7 comprehensive Jupyter notebooks demonstrating all features:

| Notebook | Topic | Difficulty | Duration |
|----------|-------|------------|----------|
| [01_fokker_planck_intro](examples/01_fokker_planck_intro.ipynb) | FP PDE basics (1D/2D) | Beginner | 5-10 min |
| [02_sde_paths_intro](examples/02_sde_paths_intro.ipynb) | SDE path simulation | Beginner | 10-15 min |
| [03_boundary_conditions](examples/03_boundary_conditions.ipynb) | BC types & effects | Intermediate | 10-15 min |
| [04_parameter_sweep_analysis](examples/04_parameter_sweep_analysis.ipynb) | Convergence & stability | Advanced | 15-20 min |
| [05_fp_vs_sde_comparison](examples/05_fp_vs_sde_comparison.ipynb) | Method comparison | Advanced | 20-30 min |
| [06_deterministic_solver_intro](examples/06_deterministic_solver_intro.ipynb) | Pure advection solver | Intermediate | 10-15 min |
| [07_chang_cooper_phi](examples/07_chang_cooper_phi.ipynb) | Chang–Cooper periodic drift–diffusion (FP vs SDE) | Intermediate | 10-15 min |

**Recommended Learning Path:**
1. Start with `01_fokker_planck_intro` for PDE basics
2. Move to `02_sde_paths_intro` for path simulation
3. Study `03_boundary_conditions` for BC understanding
4. Advance to `04_parameter_sweep_analysis` for convergence
5. Compare approaches in `05_fp_vs_sde_comparison`

See [examples/README.md](examples/README.md) for detailed notebook descriptions and usage patterns.

## Core API

### Grid Setup

```python
from stochlib.setup import Grid

# 1D grid
grid = Grid(x_start=0.0, x_end=1.0, num_points_x=201)

# 2D grid
grid = Grid(
    x_start=0.0, x_end=1.0, num_points_x=201,
    y_start=0.0, y_end=1.0, num_points_y=201
)
```

### Initial Conditions

```python
from stochlib.setup import InitialCondition

# Gaussian
ic = InitialCondition(grid, func_type="gaussian", x0=0.5, sigma_x=0.1)

# Uniform
ic = InitialCondition(grid, func_type="uniform", normalize=True)

# Custom function
def my_func(X):
    return np.exp(-X**2)
ic = InitialCondition(grid, func_type="custom", func=my_func, normalize=True)
```

### Boundary Conditions

```python
from stochlib.boundary_conditions import BoundaryConditions

# Supported types: 'periodic', 'noflux', 'open'
bc = BoundaryConditions(grid, bc_x='periodic')
bc = BoundaryConditions(grid, bc_x='noflux', bc_y='open')  # 2D
```

### Physics Configuration

```python
from stochlib.setup import VelocitiesConfig, DiffusionConfig

# Constant drift and diffusion
velocities = VelocitiesConfig(grid, mu_x=0.5)
diffusions = DiffusionConfig(grid, axes=['x'], constants={'x': 0.1})

# Space-dependent
def mu(X, t):
    return 0.5 * X

def D(X, t):
    return 0.1 * (1 + X**2)

velocities = VelocitiesConfig(grid, mu_x=mu)
diffusions = DiffusionConfig(grid, axes=['x'], functions={'x': D})
```

## Logging

StochLib includes a production-ready logging system:

```python
from stochlib import configure_logging, get_logger
import logging

# Configure once at program start
configure_logging(level=logging.INFO, verbose=True, log_file="sim.log")

# Get logger for your module
logger = get_logger("my_simulation")

# Use it
logger.info("Starting simulation")
logger.debug("Grid points: %d", grid.total_points)
logger.warning("CFL number exceeded: %.2f", cfl)
```

**Environment Variables:**
```bash
export STOCHLIB_LOGLEVEL=DEBUG          # Override logging level
export STOCHLIB_LOG_FILE=~/sim.log      # Enable file logging
```

See [LOGGING_GUIDE.md](docs/LOGGING_GUIDE.md) for complete documentation.

## Stability & Performance

### CFL Stability Constraint

For explicit PDE schemes, the time step must satisfy:
```
CFL = D * dt / dx² ≤ 0.5
```

StochLib automatically checks stability and can:
- **Raise an error** if `dt` is too large (default)
- **Auto-reduce `dt`** to safe values (use `handle_dt="auto_reduce"` in helper functions)

### Automatic Scheme Selection

The library automatically selects numerical schemes based on problem characteristics:

**Fokker-Planck:**
- **Chang-Cooper** (default): Stable for moderate Péclet numbers
- **Upwind CN**: High advection (Pe > 2)
- **Central CN**: Diffusion-dominated (Pe < 0.1)

**SDE:**
- **Euler-Maruyama**: General purpose, first-order
- **Milstein**: When diffusion Jacobian available, second-order

## Testing

The library includes 279 comprehensive tests covering all modules:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=stochlib --cov-report=html

# Run specific module
pytest tests/test_fokker_planck/
```

## Documentation

- **[FILE_STRUCTURE.md](docs/FILE_STRUCTURE.md)**: Complete architectural documentation
- **[LOGGING_GUIDE.md](docs/LOGGING_GUIDE.md)**: Logging usage and patterns
- **[DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md)**: Complete documentation index
- **[examples/README.md](examples/README.md)**: Detailed example guide

## Key Concepts

### Fokker-Planck Equation

The library solves the forward Fokker-Planck equation:

```
∂ρ/∂t = -∇·(μρ) + ½∇·(D∇ρ)
```

Where:
- `ρ(x,t)` is the probability density
- `μ(x,t)` is the drift/velocity field
- `D(x,t)` is the diffusion tensor

### Stochastic Differential Equations

Equivalent SDE representation:

```
dX = μ(X,t)dt + σ(X,t)dW
```

Where `D = ½σσᵀ` relates diffusion to noise intensity.

### Relationship

- **FP approach**: Evolution of probability distributions (PDE)
- **SDE approach**: Individual trajectory simulation (Monte Carlo)
- **Equivalence**: Ensemble statistics from SDE paths match FP solution

## Requirements

- Python ≥ 3.12
- NumPy
- Numba (JIT compilation)
- SciPy
- Matplotlib (visualization)
- Jupyter (examples)
- pytest (testing)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{stochlib2026,
  title={StochLib: Stochastic Simulations Library},
  author={[Your Name]},
  year={2026},
  url={[Repository URL]}
}
```

## Contact

- **Issues**: [GitHub Issues](https://github.com/jaaden02/Stochastic-Simulations/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jaaden02/Stochastic-Simulations/discussions)
- **Email**: jaaden@ethz.ch