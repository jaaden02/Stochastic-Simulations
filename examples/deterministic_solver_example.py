"""Example: Deterministic PDE solving with automatic scheme selection.

This script demonstrates how to solve deterministic advection PDEs using the
DeterministicPDESolver with automatic numerical scheme selection.
"""
import numpy as np
import matplotlib.pyplot as plt

from stochlib import Grid, InitialCondition
from stochlib.deterministic import DeterministicPDESolver, solve_deterministic_pde
from stochlib.deterministic.plotting import (
    plot_solution_evolution,
    plot_solution_snapshots,
    plot_spacetime_heatmap,
)


def main():
    """Run deterministic PDE examples."""
    
    # Setup: Create grid and initial condition
    grid = Grid(x_start=-5.0, x_end=5.0, num_points_x=128)
    ic = InitialCondition(grid=grid, func_type="gaussian", x0=0.0, sigma=0.5)
    
    # Time discretization
    t_eval = np.linspace(0, 2.0, 101)
    
    # Define drift velocity (constant rightward advection)
    drift_velocity = 1.0
    def drift(x):
        return drift_velocity
    
    print("=" * 60)
    print("Deterministic PDE Solver - Automatic Scheme Selection")
    print("=" * 60)
    
    # Example 1: Using convenience function with automatic scheme selection
    print("\n1. Solving with automatic scheme selection...")
    t, u = solve_deterministic_pde(
        grid=grid,
        ic=ic,
        t_eval=t_eval,
        drift=drift,
        scheme="auto",  # Automatically choose upwind or Lax-Wendroff
        boundary_mode="none",
        verbose=True,
    )
    
    # Plot solution evolution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    plot_solution_evolution(t, u, grid.X, ax=axes[0])
    axes[0].set_title("Evolution with Auto Scheme")
    
    plot_solution_snapshots(t, u, grid.X, times=[0, 25, 50, 75, 100], ax=axes[1])
    axes[1].set_title("Snapshots at Key Times")
    
    plt.tight_layout()
    plt.savefig("deterministic_auto_scheme.png", dpi=150)
    print(f"  → Saved: deterministic_auto_scheme.png")
    plt.close()
    
    # Example 2: Compare different schemes
    print("\n2. Comparing different numerical schemes...")
    schemes = ["upwind", "lax_wendroff"]
    
    fig, axes = plt.subplots(1, len(schemes), figsize=(14, 5))
    
    for idx, scheme in enumerate(schemes):
        print(f"   Solving with {scheme} scheme...")
        t, u = solve_deterministic_pde(
            grid=grid,
            ic=ic,
            t_eval=t_eval,
            drift=drift,
            scheme=scheme,
            boundary_mode="none",
        )
        
        plot_solution_snapshots(t, u, grid.X, ax=axes[idx])
        axes[idx].set_title(f"{scheme.replace('_', '-').title()} Scheme")
    
    plt.tight_layout()
    plt.savefig("deterministic_scheme_comparison.png", dpi=150)
    print(f"  → Saved: deterministic_scheme_comparison.png")
    plt.close()
    
    # Example 3: Using solver object for step-by-step control
    print("\n3. Step-by-step solving with solver object...")
    solver = DeterministicPDESolver(
        grid=grid,
        ic=ic,
        drift=drift,
        scheme="auto",
        boundary_mode="none",
    )
    
    # Manual stepping
    dt = 0.02
    n_steps = 50
    
    snapshots = []
    times = []
    
    for i in range(n_steps):
        if i % 10 == 0:
            snapshots.append(solver.u.copy())
            times.append(solver.t_current)
        solver.step(dt)
    
    snapshots.append(solver.u.copy())
    times.append(solver.t_current)
    
    # Plot manual stepping results
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.get_cmap('viridis')
    
    for idx, (t_snap, u_snap) in enumerate(zip(times, snapshots)):
        frac = idx / (len(times) - 1)
        ax.plot(grid.X, u_snap, color=cmap(frac), 
                label=f't={t_snap:.2f}', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Manual Step-by-Step Solving')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("deterministic_manual_steps.png", dpi=150)
    print(f"  → Saved: deterministic_manual_steps.png")
    plt.close()
    
    # Example 4: CFL condition warning
    print("\n4. Demonstrating CFL warning with large time step...")
    t_eval_large = np.linspace(0, 2.0, 11)  # Much larger dt
    
    t, u = solve_deterministic_pde(
        grid=grid,
        ic=ic,
        t_eval=t_eval_large,
        drift=drift,
        scheme="auto",
        boundary_mode="none",
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("Examples complete! Check the generated PNG files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
