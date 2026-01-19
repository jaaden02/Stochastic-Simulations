"""Quick validation script - no pytest needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing direct imports...")
try:
    from stochlib.setup import Grid, InitialCondition, DiffusionConfig, VelocitiesConfig
    from stochlib.boundary_conditions import BoundaryConditions
    print("✓ Direct imports successful (bypassing slow plotting)")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting Grid creation...")
grid = Grid(x_start=0.0, x_end=10.0, num_points_x=64)
print(f"✓ Grid created: {grid.num_points_x} points")
print(f"  Domain: [{grid.x_start}, {grid.x_end}]")
print(f"  dx = {grid.dx:.4f}")

print("\nTesting Initial Condition...")
ic = InitialCondition(grid, func_type='gaussian', x0=5.0, sigma_x=0.5)
print(f"✓ Initial condition created")
print(f"  Shape: {ic.f0.shape}")
print(f"  Sum: {ic.f0.sum():.4f}")

print("\nTesting Velocities Config...")
vel = VelocitiesConfig(grid, mu_x=0.5)
print(f"✓ Velocities configured")

print("\nTesting Diffusion Config...")
diff = DiffusionConfig(grid, constants={'x': 0.1})
print(f"✓ Diffusion configured")

print("\nTesting Boundary Conditions...")
bc = BoundaryConditions(grid, bc_x=BoundaryConditions.PERIODIC)
print(f"✓ Boundary conditions set")

print("\n✅ All basic component tests passed!")
