"""Minimal example: 1D Fokker-Planck with diagnostics and reporting.

Run with:
  - `uv pip install -e .` (once, to add the package)
  - `uv run python examples/use.py`

Optional: Install tqdm for a nice progress bar during simulation:
    uv pip install tqdm
"""

import numpy as np
import matplotlib

# Use non-interactive backend to avoid blocking/hanging in headless runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stochlib import (
    Grid,
    InitialCondition,
    DiffusionConfig,
    VelocitiesConfig,
    BoundaryConditions,
    configure_logging,
    get_logger,
)
from stochlib.fokker_planck import SimulationEngine, plotting


def main() -> None:
    # Logging: set once for the whole run
    configure_logging(verbose=False)
    log = get_logger("example")

    # 1) Grid and boundary conditions
    grid = Grid(x_start=0.0, x_end=10.0, num_points_x=128)
    bc = BoundaryConditions(grid, bc_x=BoundaryConditions.OPEN)

    # 2) Physics: velocities and diffusion
    velocities = VelocitiesConfig(grid, mu_x=-0.2)
    diffusions = DiffusionConfig(grid, constants={"x": 0.5})

    # 3) Initial condition
    ic = InitialCondition(grid, func_type="gaussian", x0=5.0, sigma_x=0.4)
    f = ic.f0.copy()
    f_init = f.copy()

    # 4) Simulation parameters
    dt = 0.005
    n_steps = 1000
    t_final = n_steps * dt
    # Set report_interval: N = report every N steps, 0 = no reports
    report_interval = 0
    save_reports = False
    report_file = "fp_simulation_report.txt"

    # 5) Run simulation (schemes & stability handled automatically by engine)
    engine = SimulationEngine(grid, velocities, diffusions, bc)
    result = engine.run(
        f0=f,
        t_array=np.linspace(0.0, t_final, n_steps + 1),
        save_interval=40,
        report_interval=report_interval,
        dt_user=dt,  # Let engine validate internally
        save_reports=save_reports,
        report_file=report_file,
    )
    final_f = result["final"]
    summary = result["summary"]

    # 6) Print summary (without verbose logging)
    print("\n" + "=" * 65)
    print(" " * 20 + "SUMMARY")
    print("=" * 65)
    for k, v in summary.items():
        print(f"  {k:25s} : {v}")
    print("=" * 65 + "\n")

    # 7) Generate plots using the plotting module
    plot_files = plotting.plot_simulation_summary(
        grid=grid,
        f_initial=f_init,
        f_final=final_f,
        history=result.get("history"),  # Pass full diagnostics history for detailed plots
        output_dir="plots",
        figname_prefix="fp_example",
    )
    print(f"Generated {len(plot_files)} plots:")
    for desc, path in plot_files.items():
        print(f"  {desc:20s} : {path}")


if __name__ == "__main__":
    print("Running minimal Fokker-Planck example...")
    main()
