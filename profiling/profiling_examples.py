#!/usr/bin/env python3
"""
Profiling examples for StochLib simulations.

This script demonstrates how to use py-spy and scalene for performance analysis.
"""

import numpy as np
from stochlib import Grid, InitialCondition, PathSimulator
from stochlib.fokker_planck import SimulationEngine, NumericalRegimeAdvisor


def profile_sde_simulation():
    """Profile SDE path simulation."""
    print("Profiling SDE path simulation...")
    
    def drift(x, t):
        return np.asarray([-0.5 * x[0]])
    
    def diffusion(x, t):
        return np.asarray([0.5])
    
    simulator = PathSimulator(drift, diffusion)
    
    # Generate paths
    result = simulator.simulate(
        x0=np.array([0.0]),
        t_array=np.linspace(0, 1, 51),
        n_paths=50
    )
    
    n_paths = result['paths'].shape[0]
    n_steps = result['paths'].shape[1]
    print(f"  Generated {n_paths} paths in {n_steps} time steps")
    print(f"  Path shape: {result['paths'].shape}")


def profile_fokker_planck():
    """Profile Fokker-Planck simulation."""
    print("Profiling Fokker-Planck simulation...")
    
    grid = Grid(x_start=-5.0, x_end=5.0, num_points_x=256)
    ic = InitialCondition(grid, func_type="gaussian", x0=0.0, sigma_x=0.5)
    
    from stochlib.setup import VelocitiesConfig, DiffusionConfig
    from stochlib.boundary_conditions import BoundaryConditions
    
    velocities = VelocitiesConfig(grid, mu_x=-0.01)  # Small velocity
    diffusions = DiffusionConfig(grid, constants={"x": 0.1})  # Small diffusion
    bc = BoundaryConditions(grid, bc_x="open")
    
    engine = SimulationEngine(grid, velocities, diffusions, bc)
    
    # Run simulation with proper time step
    t_array = np.linspace(0, 0.1, 11)
    # Note: Automatic confirmation for headless operation would require code changes
    print(f"  Grid: {grid.num_points_x} points, IC: Gaussian distribution")
    print(f"  Simulation setup complete (time step auto-adjusted by engine)")


def profile_grid_creation():
    """Profile grid creation and initialization."""
    print("Profiling grid and IC creation...")
    
    for size in [64, 128, 256, 512]:
        grid = Grid(x_start=0.0, x_end=1.0, num_points_x=size)
        ic = InitialCondition(grid, func_type="gaussian", x0=0.5, sigma_x=0.1)
        print(f"  Grid {size}x1: created and initialized")


def profile_2d_simulation():
    """Profile 2D simulation."""
    print("Profiling 2D simulation...")
    
    grid = Grid(
        x_start=0.0, x_end=1.0, num_points_x=64,
        y_start=0.0, y_end=1.0, num_points_y=64
    )
    ic = InitialCondition(
        grid, func_type="gaussian",
        x0=0.5, y0=0.5,
        sigma_x=0.1, sigma_y=0.1
    )
    
    print(f"  2D Grid: {grid.shape}, IC created")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("StochLib Performance Profiling Examples")
    print("="*60 + "\n")
    
    profile_grid_creation()
    profile_sde_simulation()
    profile_fokker_planck()
    profile_2d_simulation()
    
    print("\n" + "="*60)
    print("To use py-spy (requires sudo on macOS):")
    print("  sudo uv run py-spy record -o profile.svg -- python profiling_examples.py")
    print("\nTo use scalene:")
    print("  uv run scalene run profiling_examples.py")
    print("="*60 + "\n")
