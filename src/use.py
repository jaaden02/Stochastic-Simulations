"""Minimal example: 1D Fokker-Planck with diagnostics and reporting.

Run with: `python use.py` from this directory.
"""

import numpy as np
import matplotlib

# Use non-interactive backend to avoid blocking/hanging in headless runs
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stochlib import (
	Grid,
	InitialCondition,
	DiffusionConfig,
	VelocitiesConfig,
	BoundaryConditions,
	configure_logging,
	get_logger,
	plotting,
)
from stochlib.fokker_planck import SimulationEngine


def main() -> None:
	# Logging: set once for the whole run
	configure_logging(verbose=False)
	log = get_logger("example")

	# 1) Grid and boundary conditions
	grid = Grid(x_start=0.0, x_end=10.0, num_points_x=128)
	bc = BoundaryConditions(grid, bc_x=BoundaryConditions.OPEN)

	# 2) Physics: velocities and diffusion
	velocities = VelocitiesConfig(grid, mu_x=-0.2)
	diffusions = DiffusionConfig(grid, constants={'x': 0.5})

	# 3) Initial condition
	ic = InitialCondition(grid, func_type='gaussian', x0=5.0, sigma_x=0.4)
	f = ic.f0.copy()
	f_init = f.copy()

	# 4) Simulation parameters
	dt = 0.005
	n_steps = 200
	t_final = n_steps * dt
	report_interval = 20
	save_reports = True
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
	final_f = result['final']
	summary = result['summary']
	log.info("Summary:")
	for k, v in summary.items():
		log.info("  %s: %s", k, v)

	# 6) Generate plots using the plotting module
	plot_files = plotting.plot_simulation_summary(
		grid=grid,
		f_initial=f_init,
		f_final=final_f,
		history=result.get('history'),  # Pass full diagnostics history for detailed plots
		output_dir="plots",
		figname_prefix="fp_example",
	)
	log.info("Generated %d plots:", len(plot_files))
	for desc, path in plot_files.items():
		log.info("  %s: %s", desc, path)


if __name__ == "__main__":
	print("Running minimal Fokker-Planck example...")
	main()
	main()
