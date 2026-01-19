# StochLib File Structure

Comprehensive documentation of the stochlib package directory organization and module structure.

## Project Root Structure

```
stochastic-simulations/
├── src/stochlib/                 # Main package source code
├── tests/                        # Test suite (pytest)
├── examples/                     # Example scripts demonstrating functionality
├── pyproject.toml               # Project metadata and dependencies
├── README.md                     # Project overview
├── FILE_STRUCTURE.md            # This file
└── .gitignore                    # Git ignore rules
```

## Source Package Structure (`src/stochlib/`)

The stochlib package is organized into three main submodules plus core utilities:

```
src/stochlib/
├── __init__.py                   # Package root, public API exports
├── py.typed                      # PEP 561 marker for type hints
├── setup.py                      # Core configuration classes
├── boundary_conditions.py         # Boundary condition handlers
├── logging_utils.py             # Centralized logging utilities
│
├── fokker_planck/               # Fokker-Planck PDE solver module
│   ├── __init__.py
│   ├── solver.py               # FokkerPlanckSolver class
│   ├── diagnostics.py          # StabilityAnalyzer, SolutionDiagnostics
│   ├── selector.py             # NumericalRegimeAdvisor, SimulationEngine
│   ├── kernel.py               # Low-level JIT-compiled stepping kernels
│   ├── plotting/
│   │   ├── __init__.py
│   │   ├── distributions.py    # 1D/2D distribution plotting
│   │   ├── diagnostics.py      # Diagnostic metric plots
│   │   └── utils.py            # High-level plotting utilities
│   └── test_dependencies.py    # (internal)
│
├── sde/                         # Stochastic Differential Equation solver module
│   ├── __init__.py
│   ├── solver.py               # PathSimulator class
│   ├── kernel.py               # Euler-Maruyama, Milstein kernels
│   ├── selector.py             # StepSchemeAdvisor for scheme selection
│   ├── diagnostic.py           # PathDiagnostics for statistics tracking
│   ├── sampling.py             # Initial condition sampling utilities
│   ├── comparison.py           # Distribution/moment comparison tools
│   ├── bridge.py               # Fokker-Planck to SDE parameter conversion
│   └── plotting/
│       ├── __init__.py
│       ├── trajectories.py     # Path trajectory and mean/variance plots
│       ├── phase_portrait.py   # 2D phase space visualization
│       └── paths.py            # Individual path plots with threshold sampling
│
└── deterministic/              # Deterministic PDE solver module
    ├── __init__.py
    ├── solver.py               # DeterministicPDESolver class
    ├── diagnostics.py          # Solution analysis and validation tools
    └── plotting/
        ├── __init__.py
        ├── evolution.py        # Solution evolution visualization
        ├── diagnostic_plots.py  # Diagnostic metric displays
        └── comparison.py       # Numerical vs. exact solution comparison
```

## Module Descriptions

### Core Modules

#### `setup.py`
Defines fundamental configuration classes:
- **Grid**: Spatial discretization for 1D, 2D, 3D domains
- **InitialCondition**: Initial probability distributions
- **DiffusionConfig**: Diffusion coefficient specifications
- **VelocitiesConfig**: Velocity/drift field specifications
- **SimulationSetup**: Complete simulation configuration container

#### `boundary_conditions.py`
Implements boundary condition management:
- Supports 'open', 'noflux', 'periodic', 'reflect', 'absorb' modes
- Per-axis specification for multi-dimensional grids
- Applied consistently across all solvers

#### `logging_utils.py`
Centralized logging infrastructure:
- **configure_logging()**: Set up logging with console and optional file handlers
- **get_logger()**: Get named loggers under the stochlib namespace
- **log_performance()**: Decorator for timing and error tracking
- Environment variables: `STOCHLIB_LOGLEVEL`, `STOCHLIB_LOG_FILE`

### Fokker-Planck Module (`fokker_planck/`)

**Purpose**: Solve the Fokker-Planck PDE: ∂f/∂t = -∇·(vf) + ∇²(Df)

**Key Classes**:
- `FokkerPlanckSolver`: Core PDE solver using finite difference schemes
- `StabilityAnalyzer`: Von Neumann stability analysis
- `SolutionDiagnostics`: Quality metrics (mass, entropy, convergence)
- `NumericalRegimeAdvisor`: Automatic scheme selection
- `SimulationEngine`: High-level simulation orchestration

**Plotting Functions**:
- `plot_1d_distribution()`: 1D probability density plots
- `plot_2d_distribution()`: 2D heatmap visualization
- `plot_mass_conservation()`: Mass conservation tracking
- `plot_entropy()`: Entropy evolution
- `plot_convergence()`: Grid convergence analysis

### SDE Module (`sde/`)

**Purpose**: Simulate sample paths for SDEs: dX_t = μ(X,t)dt + σ(X,t)dW_t

**Key Classes**:
- `PathSimulator`: Many-path SDE simulation
- `StepSchemeAdvisor`: Euler-Maruyama vs. Milstein selection
- `PathDiagnostics`: Running statistics accumulation

**Utilities**:
- `euler_maruyama_step()`: EM stepping kernel
- `milstein_step()`: Milstein stepping kernel (requires diffusion Jacobian)
- `sample_from_distribution()`: Initial condition sampling
- `paths_to_histogram()`: Path ensemble statistics
- `fp_to_sde_drift/diffusion()`: Parameter conversion from FP to SDE

**Plotting Functions**:
- `plot_trajectories()`: Individual path visualization
- `plot_mean_variance()`: Mean ± std bands
- `plot_phase_portrait()`: 2D phase space projection
- `plot_paths_with_threshold()`: Smart sampling for large path ensembles

### Deterministic Module (`deterministic/`)

**Purpose**: Solve deterministic advection PDEs: ∂u/∂t + ∂(vu)/∂x = 0

**Key Classes**:
- `DeterministicPDESolver`: Advection solver with auto scheme selection
- `DeterministicDiagnostics`: Solution quality tracking
- Schemes: Upwind (1st order), Lax-Wendroff (2nd order), Beam-Warming (2nd order upwind)

**Diagnostics Functions**:
- `check_mass_conservation()`: Mass preservation validation
- `compute_solution_moments()`: Mean, variance, skewness
- `compute_cfl_number()`: CFL stability criterion
- `check_positivity()`: Physical constraint verification
- `compute_total_variation()`: TVD scheme validation

**Plotting Functions**:
- `plot_solution_evolution()`: Multi-snapshot time evolution
- `plot_spacetime_heatmap()`: Space-time diagram
- `plot_diagnostics()`: Comprehensive diagnostic dashboard
- `plot_comparison_with_exact()`: Numerical vs. analytical comparison

## Public API

The package main namespace (`stochlib.__init__.py`) exports:

### Core Classes
```python
from stochlib import (
    Grid,
    InitialCondition,
    DiffusionConfig,
    VelocitiesConfig,
    SimulationSetup,
    BoundaryConditions,
)
```

### Solvers & Tools
```python
from stochlib import (
    FokkerPlanckSolver,      # FP PDE solver
    PathSimulator,            # SDE path simulator
    DeterministicPDESolver,   # Advection PDE solver
    StepSchemeAdvisor,        # SDE scheme selection
    PathDiagnostics,          # SDE statistics tracking
)
```

### Utilities
```python
from stochlib import (
    configure_logging,
    get_logger,
)
```

### Submodule Namespaces
```python
import stochlib.fokker_planck
import stochlib.fokker_planck.plotting

import stochlib.sde
import stochlib.sde.plotting

import stochlib.deterministic
import stochlib.deterministic.plotting
```

## Test Structure (`tests/`)

Test files follow pytest conventions:
- `test_*.py`: Test modules
- `conftest.py`: Shared fixtures and configuration
- Test organization mirrors source structure

**Key Test Files**:
- `test_grid.py`: Grid discretization
- `test_initial_condition.py`: Initial condition setup
- `test_boundary_conditions.py`: BC implementations
- `test_numerical_schemes.py`: FP solver schemes
- `test_path_simulator.py`: SDE path generation
- `test_simulation_engine.py`: High-level orchestration
- `test_plotting.py`: Visualization functions
- `test_integration.py`: End-to-end workflows
- `test_stability.py`: Stability analysis

## Example Scripts (`examples/`)

Demonstration scripts showing practical usage:
- `deterministic_solver_example.py`: Auto scheme selection for advection PDEs
- Additional examples for FP and SDE solvers

## Configuration Files

### `pyproject.toml`
Project metadata, dependencies, and tool configuration:
- Build system (setuptools)
- Dependencies (NumPy, SciPy, Matplotlib, Numba)
- pytest configuration with custom markers
- Logging configuration

### `.gitignore`
Excludes:
- Python artifacts (__pycache__, .pyc, etc.)
- Virtual environments
- IDE files (.vscode, .idea)
- Generated plots (*.png, *.jpg)
- Test coverage files
- Jupyter notebooks (.ipynb_checkpoints)

## Logging Configuration

The package uses Python's standard `logging` module with a centralized setup:

```python
from stochlib import configure_logging, get_logger

# Configure root logger
configure_logging(level=logging.DEBUG, verbose=True, log_file="sim.log")

# Get named loggers for modules
logger = get_logger("my_module")
logger.info("Simulation started")
```

**Environment Variables**:
- `STOCHLIB_LOGLEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `STOCHLIB_LOG_FILE`: Path to log file

## Development Notes

### Type Hints
All modules use type hints following PEP 484 conventions. The package is marked as `py.typed` for static type checking support.

### Docstrings
All public functions and classes use Google-style docstrings with:
- One-line summary
- Detailed description
- Parameters section
- Returns section
- Examples section (where applicable)
- Notes section (for implementation details)

### Code Style
- Follows PEP 8 with 88-character line length target
- Uses snake_case for functions/variables, PascalCase for classes
- UPPER_CASE for module-level constants

### Performance
- Hot-path kernels use Numba JIT compilation (@njit decorator)
- NumPy vectorization throughout
- Grid-based vectorized operations preferred over loops

### Testing
- 279 tests covering all major components
- Test-driven development with pytest
- Both unit tests and integration tests
- Performance benchmarks in select tests
