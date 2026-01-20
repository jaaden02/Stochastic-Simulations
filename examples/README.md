# Stochastic Simulations Examples

This directory contains comprehensive Jupyter notebooks demonstrating various features and capabilities of the stochlib library.

## Quick Start

All notebooks require:
- `stochlib` (installed from source)
- `numpy`, `matplotlib`

To run locally, ensure the repository root is in your Python path, then open any notebook with Jupyter:
```bash
jupyter notebook 01_fokker_planck_intro.ipynb
```

## Notebooks

All notebooks follow a consistent structure with setup, examples, diagnostics, and visualization.

### [01_fokker_planck_intro.ipynb](01_fokker_planck_intro.ipynb)
**Introduction to Fokker-Planck Equation Solver**

Demonstrates the basic Fokker-Planck PDE solver for diffusion and advection-diffusion problems.

**Topics Covered:**
- 1D pure diffusion with periodic boundary conditions
- 2D advection-diffusion problems
- Solution diagnostics (mass, entropy)
- Visualization of probability distributions

**Key Classes:**
- `Grid`: Spatial domain discretization
- `InitialCondition`: Setup initial probability distribution
- `BoundaryConditions`: Boundary condition specification
- `FokkerPlanckSolver`: Main PDE solver
- `SolutionDiagnostics`: Compute solution properties

**Learning Outcome:** Understand how to set up and solve Fokker-Planck equations with various configurations.

---

### [02_sde_paths_intro.ipynb](02_sde_paths_intro.ipynb)
**Stochastic Differential Equation (SDE) Path Simulation**

Demonstrates Monte Carlo path simulation for SDEs using different numerical schemes.

**Topics Covered:**
- Simple Brownian motion (dX = dW)
- Advection-diffusion (dX = μ(X)dt + σ(X)dW)
- Automatic scheme selection (Euler-Maruyama vs Milstein)
- Path ensemble statistics (mean, variance)
- Trajectory visualization and diagnostics

**Key Classes:**
- `PathSimulator`: Monte Carlo SDE solver
- `StepSchemeAdvisor`: Automatic scheme recommendation
- `PathDiagnostics`: Compute path ensemble properties
- `plot_trajectories`, `plot_mean_variance`: Visualization functions

**Learning Outcome:** Learn to simulate and analyze stochastic processes, compare numerical schemes, and extract statistical properties from path ensembles.

---

### [03_boundary_conditions.ipynb](03_boundary_conditions.ipynb)
**Boundary Condition Types and Effects**

Comprehensive comparison of all available boundary condition types for FP equations.

**Topics Covered:**
- **Periodic BC**: Cyclic domain, solution wraps around
- **No-Flux BC**: Reflective, probability confined to domain
- **Absorbing BC**: Probability exits at boundaries
- **Reflecting BC**: Hard wall bouncing at boundaries
- Mass conservation properties for each BC type
- Visualization of qualitative differences

**Key Concepts:**
- How boundary conditions affect solution evolution
- Mass conservation vs absorption/reflection
- Choosing appropriate BCs for physical problems

**Learning Outcome:** Understand boundary condition selection and how they affect solution behavior in physical systems.

---

### [04_parameter_sweep_analysis.ipynb](04_parameter_sweep_analysis.ipynb)
**Numerical Convergence and Parameter Sensitivity**

Analysis of numerical convergence properties and parameter effects.

**Topics Covered:**
- **Diffusion Coefficient Sensitivity**: How diffusion rate affects spreading
- **Grid Resolution Convergence**: Accuracy vs computational cost
- **Time Stepping**: Effect of temporal resolution
- **Courant-Friedrichs-Lewy (CFL) Stability**: dt/dx² constraints
- Convergence rate estimation
- Practical guidance for parameter selection

**Key Analysis:**
- Variance growth follows theoretical 2Dt for pure diffusion
- Grid convergence typically 2nd order
- CFL condition: D*dt/dx² ≤ 0.5 for stability
- Trade-offs between accuracy and computational cost

**Learning Outcome:** Learn to select appropriate numerical parameters and understand fundamental stability and convergence constraints.

---

### [05_fp_vs_sde_comparison.ipynb](05_fp_vs_sde_comparison.ipynb)
**Fokker-Planck vs SDE Path Comparison**

Compares two equivalent approaches: solving the FP PDE vs Monte Carlo SDE simulation.

**Topics Covered:**
- FP to SDE conversion using bridge functions
- Path simulation with boundary modes (absorbing, reflecting)
- Building histograms from path ensembles
- Distribution and moment comparison
- Visual comparison of methods

**Key Functions:**
- `fp_to_sde_drift`, `fp_to_sde_diffusion`: Bridge conversion
- `paths_to_histogram`: Convert paths to distribution
- `compare_distributions`, `compare_moments`: Quantitative comparison

**Learning Outcome:** Understand equivalence between PDE and stochastic approaches, verify numerical methods against each other.

---

### [06_deterministic_solver_intro.ipynb](06_deterministic_solver_intro.ipynb)
**Deterministic Advection PDE Solver**

Demonstrates deterministic advection equation solving with automatic numerical scheme selection.

**Topics Covered:**
- Pure advection (linear transport) equations
- Automatic scheme selection based on CFL number
- Comparison of upwind vs Lax-Wendroff schemes
- Manual step-by-step integration
- CFL stability analysis

**Key Classes:**
- `DeterministicPDESolver`: Solves advection PDEs
- `solve_deterministic_pde`: Convenience function
- Automatic scheme advisor

**Key Concepts:**
- CFL Condition: `c·dt/dx ≤ 1` for stability
- Upwind scheme: Stable but diffusive
- Lax-Wendroff: Less diffusive, oscillates if CFL > 1

**Learning Outcome:** Understand deterministic PDE solving, numerical stability constraints, and scheme selection.

---

### [07_chang_cooper_phi.ipynb](07_chang_cooper_phi.ipynb)
**Chang–Cooper Drift–Diffusion (Periodic φ) with FP vs SDE**

Demonstrates the Chang–Cooper scheme on a periodic angular variable and compares the FP solution to SDE path histograms.

**Topics Covered:**
- 1D periodic domain using Chang–Cooper weighting
- Time-dependent drift from the original 3D model reduced to φ
- Stability check with `StabilityAnalyzer`
- FP vs SDE histogram comparison and wrapping on the circle

**Key Classes:**
- `Grid`, `InitialCondition`, `VelocitiesConfig`, `DiffusionConfig`
- `BoundaryConditions` (periodic)
- `SimulationEngine` (FP) and `PathSimulator` (SDE)
- `fp_to_sde_drift`, `fp_to_sde_diffusion` for FP→SDE bridging

**Learning Outcome:** See how Chang–Cooper preserves positivity on a periodic domain and how the FP solution aligns with SDE paths.

---

## Running the Notebooks

### Setup
1. Install dependencies:
   ```bash
   pip install numpy matplotlib jupyter
   ```

2. From the repository root, start Jupyter:
   ```bash
   jupyter notebook examples/
   ```

### Execution Tips
- Run notebooks in order (01 → 07) for progressive learning
- Each notebook is self-contained and can run independently
- Notebooks configure logging to INFO level for traceability
- Some notebooks (especially 04) may take a few minutes to complete parameter sweeps

## Recommended Learning Path

**Beginner (2-3 hours)**:
1. Start with 01: Fokker-Planck PDE basics
2. Continue to 02: SDE path simulation alternative
3. Explore 03: Boundary conditions effects

**Intermediate (4-5 hours)**:
- Complete beginner path
- 04: Numerical convergence and stability
- 05: Compare FP vs SDE approaches

**Advanced (6+ hours)**:
- Complete intermediate path
- 06: Deterministic advection and CFL analysis
- 07: Chang–Cooper periodic drift–diffusion (FP vs SDE)
- Study source code patterns

## Common Patterns

### Creating a Grid
```python
from stochlib.setup import Grid
grid = Grid(bounds=[(0.0, 1.0)], npoints=[101])  # 1D: 101 points in [0,1]
```

### Defining Initial Conditions
```python
from stochlib.setup import InitialCondition
ic = InitialCondition(
    shape=lambda x: (1.0 / np.sqrt(2 * np.pi * 0.1)) * np.exp(-0.5 * x[0]**2 / 0.1)
)
```

### Setting Boundary Conditions
```python
from stochlib.setup import BoundaryConditions
bc = BoundaryConditions(bcs=[("periodic", "periodic")])  # for 1D
```

### Solving Fokker-Planck
```python
from stochlib.fokker_planck import FokkerPlanckSolver
solver = FokkerPlanckSolver(grid=grid, initial_condition=ic, 
                            boundary_conditions=bc, ...)
result = solver.solve(t_array=np.linspace(0, 1, 101))
```

### Simulating SDE Paths
```python
from stochlib.sde import PathSimulator
sim = PathSimulator(drift=drift_func, diffusion=diffusion_func, scheme='auto')
result = sim.simulate(x0=initial_positions, t_array=times, save_paths=True)
```

## Troubleshooting

**Import Error: stochlib not found**
- Ensure repository root is in Python path (notebooks handle this automatically)
- Check `from stochlib import ...` imports work in terminal

**Visualization not displaying**
- Ensure matplotlib is installed
- In Jupyter, run `%matplotlib inline` if plots don't show

**Slow execution**
- Parameter sweep notebooks (04) compute many solutions—this is normal
- Grid refinement studies require multiple solves—time depends on grid size
- Consider reducing number of time steps for faster preview

## Contributing

To add new examples:
1. Follow the naming convention: `0X_topic_name.ipynb`
2. Include markdown descriptions of what's being demonstrated
3. Add explanatory comments in code cells
4. Include visualization and interpretation
5. Update this README with new entry

## References

See the main project README for mathematical background and theoretical references.
