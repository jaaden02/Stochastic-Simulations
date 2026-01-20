"""Numerical regime advisor and simulation engine for Fokker-Planck equations.

Provides automatic scheme selection based on problem characteristics and a high-level
simulation engine for running Fokker-Planck PDE solutions.
"""

from typing import Optional, Dict, Any
import numpy as np
import logging
import time

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
from ..setup import Grid
from .solver import FokkerPlanckSolver
from ..logging_utils import get_logger
from .diagnostics import StabilityAnalyzer, SolutionDiagnostics

logger = get_logger("fokker_planck.selector")


class NumericalRegimeAdvisor:
    """Analyzes the physics to select the best numerical strategy."""

    @staticmethod
    def recommend_schemes(
        grid: Grid,
        velocities,
        diffusions,
        t: Optional[float] = 0.0,
    ) -> Dict[str, str]:
        """Recommend numerical schemes based on Peclet number analysis.

        Parameters
        ----------
        grid : Grid
            Spatial grid configuration
        velocities : VelocitiesConfig
            Advection velocity configuration
        diffusions : DiffusionConfig
            Diffusion coefficient configuration
        t : float, optional
            Current time (for time-dependent coefficients)

        Returns
        -------
        dict
            Mapping of axis names to scheme names ('upwind_cn', 'chang_cooper', 'central_cn')
        """
        # 1. Build the fields at the starting time
        mu_fields: Dict = velocities.build_fields(t=t, evaluate=True)
        d_fields: Dict = diffusions.build_fields(t=t, evaluate=True)
        logger.debug("Recommend schemes: built fields at t=%.3e", t)

        recommendations: Dict[str, str] = {}
        deltas: Dict[str, float] = grid.deltas
        for ax in grid.axis_names:
            dx: float = deltas[ax]
            max_mu: float = np.max(np.abs(mu_fields[ax]))
            max_d: float = np.max(d_fields[ax])

            # 2. Calculate the Grid Peclet Number: Pe = (|v| * dx) / D
            pe_max: float = (max_mu * dx) / (max_d + 1e-16)

            logger.debug(
                "Axis %s: dx=%.3e max_mu=%.3e max_d=%.3e Pe=%.3e",
                ax,
                dx,
                max_mu,
                max_d,
                pe_max,
            )

            # 3. Heuristic Selection
            scheme_choice: str
            if max_d < 1e-14:
                # Pure advection requires the stability of Upwind
                scheme_choice = "upwind_cn"
            elif pe_max > 2.0:
                # High-speed flow: Central differences would oscillate
                scheme_choice = "upwind_cn"
            elif 0.1 < pe_max <= 2.0:
                # The "Golden Zone" for Chang-Cooper stability
                scheme_choice = "chang_cooper"
            else:
                # Highly diffusive: use 2nd-order Central accuracy
                scheme_choice = "central_cn"

            recommendations[ax] = scheme_choice

        return recommendations


class SimulationEngine:
    """Orchestrates the setup, validation, and execution of the simulation."""

    def __init__(self, grid: Grid, velocities, diffusions, bc_manager) -> None:
        self.grid: Grid = grid
        self.velocities = velocities
        self.diffusions = diffusions
        self.bc_manager = bc_manager

    def run(
        self,
        f0: np.ndarray,
        t_array: np.ndarray,
        save_interval: int = 10,
        diagnostics: Optional[SolutionDiagnostics] = None,
        report_interval: int = 10,
        mass_tolerance: float = 1e-3,
        confirm_run: bool = True,
        save_reports: bool = False,
        report_file: Optional[str] = None,
        dt_user: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run the Fokker-Planck simulation.

        Parameters
        ----------
        f0 : ndarray
            Initial probability/distribution field
        t_array : ndarray
            Array of time points
        save_interval : int
            Save output every N steps (snapshots)
        diagnostics : SolutionDiagnostics, optional
            Custom diagnostics instance; if None, a default is created.
        report_interval : int
            Diagnostics reporting cadence when diagnostics is created internally.
        mass_tolerance : float
            Mass change tolerance for diagnostics when created internally.
        confirm_run : bool
            If True, show pre-run report and ask for confirmation before executing.
        save_reports : bool
            If True, save all reports (pre-run and final) to a file. Default: False.
        report_file : str, optional
            Output file path for reports. If None and save_reports=True,
            defaults to 'simulation_report.txt'.
        dt_user : float, optional
            User-specified timestep. If provided, used for validation. If None,
            automatically computed from t_array spacing.

        Returns
        -------
        dict
            {"final": ndarray, "snapshots": list, "times": list, "summary": dict, "reports": str}
        """
        # A. THE DETERMINATION PHASE
        # Automatically choose schemes based on physics
        auto_schemes: Dict[str, str] = NumericalRegimeAdvisor.recommend_schemes(
            self.grid, self.velocities, self.diffusions, t=t_array[0]
        )
        logger.info("Engine Decision: %s", auto_schemes)
        logger.debug("Run config: save_interval=%d", save_interval)

        # B. THE VALIDATION PHASE
        # Use a separate analyzer to check if the user's dt is safe
        if dt_user is None:
            dt_user: float = t_array[1] - t_array[0] if len(t_array) > 1 else 0.01
        logger.debug("Computed dt_user=%.3e from t_array", dt_user)
        report = StabilityAnalyzer.analyze(
            self.grid, self.velocities, self.diffusions, dt_user, t=t_array[0]
        )
        # Pre-run cost estimate
        num_steps = max(1, len(t_array) - 1)
        per_array_gb = report.memory_gb
        num_arrays = 1 + 2 * len(self.grid.axis_names)  # f + MU + D per active axis
        est_mem_gb = per_array_gb * num_arrays
        # Crude runtime estimate: ops per cell per step ~ 80
        ops_per_cell_step = 80
        est_ops = ops_per_cell_step * self.grid.total_points * num_steps
        assumed_perf = 1e9  # ops/sec rough guess
        est_seconds = est_ops / assumed_perf
        logger.debug(
            "Estimates: num_arrays=%d per_array_gb=%.3f ops=%.3e assumed_perf=%.3e",
            num_arrays,
            per_array_gb,
            est_ops,
            assumed_perf,
        )

        # Enhanced pre-run report with full setup
        pre_lines = [
            "=" * 65,
            " " * 15 + "PRE-RUN SIMULATION REPORT",
            "=" * 65,
            "",
            "GRID CONFIGURATION:",
            f"  Active axes          : {', '.join(self.grid.axis_names)}",
            f"  Total grid points    : {self.grid.total_points:,}",
            f"  Grid deltas (Δ)      : {self.grid.deltas}",
            f"  Volume element (dV)  : {self.grid.volume_element:.3e}",
            "",
            "BOUNDARY CONDITIONS:",
        ]
        for ax in self.grid.axis_names:
            bc_type = self.bc_manager.for_axis(ax)
            pre_lines.append(f"  {ax:6s} : {bc_type}")

        pre_lines.extend(
            [
                "",
                "PHYSICS CONFIGURATION:",
                f"  Max velocity (|μ|)   : {self.velocities.max_velocity_magnitude(t=t_array[0]):.3e}",
                f"  Max diffusion (D)    : {np.max([np.max(d) for d in self.diffusions.build_fields(t=t_array[0], evaluate=True).values()]):.3e}",
                "",
                "TIME STEPPING:",
                f"  dt (requested)       : {dt_user:.3e}",
                f"  dt_max (CFL stable)  : {report.dt_max:.3e}",
                f"  Total steps          : {num_steps}",
                f"  Time window          : [t={t_array[0]:.3e}, t={t_array[-1]:.3e}]",
                "",
                "NUMERICAL SCHEMES:",
            ]
        )
        for ax, scheme in auto_schemes.items():
            pre_lines.append(f"  {ax:6s} : {scheme}")

        pre_lines.extend(
            [
                "",
                "DIAGNOSTICS:",
                f"  Report interval      : every {report_interval} step(s)",
                f"  Mass tolerance       : {mass_tolerance:.2e}",
                "",
                "RESOURCE ESTIMATES:",
                f"  Per-field memory     : {per_array_gb:.4f} GB",
                f"  Num fields           : {num_arrays} (f + velocities + diffusion)",
                f"  Est. total memory    : {est_mem_gb:.4f} GB",
                f"  Est. runtime         : {est_seconds:.4f} s (rough; peak may vary)",
                "=" * 65,
            ]
        )

        pre_report = "\n".join(pre_lines)
        print("\n" + pre_report)

        # Store reports for file saving
        all_reports = [pre_report + "\n"]

        if confirm_run:
            import sys

            if sys.stdin is not None and sys.stdin.isatty():
                resp = input("Proceed with run? [y/N]: ").strip().lower()
                if resp not in ("y", "yes"):
                    raise RuntimeError("Run aborted by user after pre-run report")
            else:
                logger.info("No TTY available; proceeding without prompt.")

        if not report.is_stable:
            raise ValueError(
                f"Stability Error: dt={dt_user:.2e} exceeds dt_max={report.dt_max:.2e}. "
                f"Solutions: "
                f"(1) Reduce dt (e.g., dt={report.dt_max*0.9:.2e}), "
                f"(2) Use finer grid (more grid points), "
                f"(3) Reduce drift/diffusion magnitude. "
                f"See DEBUGGING.md for stability analysis."
            )
        logger.debug("Stability OK: dt=%.3e (dt_max=%.3e)", dt_user, report.dt_max)

        # C. THE EXECUTION PHASE
        # Initialize the high-performance solver

        solver = FokkerPlanckSolver(self.bc_manager, schemes=auto_schemes)

        if diagnostics is None:
            diagnostics = SolutionDiagnostics(
                self.grid,
                volume_element=self.grid.volume_element,
                boundary_conditions=self.bc_manager,
                report_interval=report_interval,
                mass_tolerance=mass_tolerance,
            )
        else:
            logger.debug("Using provided diagnostics instance")

        return self._time_loop(
            f0,
            t_array,
            solver,
            save_interval,
            diagnostics,
            all_reports,
            save_reports,
            report_file,
        )

    def _time_loop(
        self,
        f: np.ndarray,
        t_array: np.ndarray,
        solver,
        save_interval: int,
        diagnostics: SolutionDiagnostics,
        all_reports: list,
        save_reports: bool,
        report_file: Optional[str],
    ) -> Dict[str, Any]:
        """Execute the time-stepping loop.

        Parameters
        ----------
        f : ndarray
            Initial field
        t_array : ndarray
            Time points
        solver : FokkerPlanckSolver
            Numerical solver instance
        save_interval : int
            Output save frequency
        diagnostics : SolutionDiagnostics
            Diagnostics tracker instance
        all_reports : list
            Accumulator for all report strings (pre-run report already included)
        save_reports : bool
            Whether to save reports to a file
        report_file : str, optional
            File path for saving reports

        Returns
        -------
        dict
            {"final": ndarray, "snapshots": list, "times": list, "summary": dict, "report_text": str}
        """

        snapshots = []
        times = []
        f_curr = f.copy()
        logger.debug("Starting time loop with %d steps", len(t_array) - 1)

        # Track total runtime
        start_time = time.time()

        # Setup progress bar if available
        iterator = (
            tqdm(
                range(1, len(t_array)),
                desc="Simulating",
                unit="step",
                disable=not HAS_TQDM,
            )
            if HAS_TQDM
            else range(1, len(t_array))
        )

        for idx in iterator:
            t_prev = t_array[idx - 1]
            t_curr = t_array[idx]
            dt = t_curr - t_prev
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Step %d/%d: t_prev=%.3e t_curr=%.3e dt=%.3e",
                    idx,
                    len(t_array) - 1,
                    t_prev,
                    t_curr,
                    dt,
                )
            mu_fields = self.velocities.build_fields(t=t_prev, evaluate=True)
            d_fields = self.diffusions.build_fields(t=t_prev, evaluate=True)
            f_curr = solver.solve_step(f_curr, mu_fields, d_fields, dt)

            metrics = diagnostics.analyze_step(f_curr, t_curr)
            metrics["time"] = t_curr
            if metrics.get("warnings"):
                logger.warning("t=%.3e warnings=%s", t_curr, metrics["warnings"])

            if (idx % save_interval) == 0:
                snapshots.append(f_curr.copy())
                times.append(t_curr)
                logger.debug("Saved snapshot at step %d (t=%.3e)", idx, t_curr)

            # Update progress bar with current time and mass
            if HAS_TQDM:
                iterator.set_postfix(
                    {"t": f"{t_curr:.4f}", "mass": f"{metrics['mass']:.3e}"},
                    refresh=True,
                )

        elapsed_time = time.time() - start_time
        summary = diagnostics.summary()

        # Build final report with runtime
        final_lines = [
            "",
            "=" * 65,
            " " * 20 + "FINAL REPORT",
            "=" * 65,
            "",
            "SIMULATION COMPLETION SUMMARY:",
            f"  Total runtime        : {elapsed_time:.4f} s",
            f"  Total steps completed: {summary['num_steps']}",
            f"  Time window          : [t={t_array[0]:.3e}, t={t_array[-1]:.3e}]",
            "",
            "SOLUTION STATISTICS:",
            f"  Total mass change    : {summary['total_mass_change']:.3e}",
            f"  Mean mass            : {summary['mean_mass']:.6e}",
            f"  Min solution value   : {summary['min_solution_value']:.3e}",
            f"  Max solution value   : {summary['max_solution_value']:.3e}",
            "",
            "ENTROPY:",
            f"  Initial entropy      : {summary['entropy_start']:.6e}",
            f"  Final entropy        : {summary['entropy_end']:.6e}",
            f"  Monotonically incr.  : {summary['entropy_is_monotonic']}",
            "",
            "QUALITY FLAGS:",
            f"  Any negatives        : {summary['any_negative']}",
            f"  Total warnings       : {summary['total_warnings']}",
            f"  Critical warnings    : {len(summary['critical_warnings'])}",
            "=" * 65,
        ]

        final_report = "\n".join(final_lines)
        print("\n" + final_report)
        all_reports.append(final_report)

        # Save reports to file if requested
        full_report_text = "\n".join(all_reports)
        if save_reports:
            if report_file is None:
                report_file = "simulation_report.txt"
            try:
                with open(report_file, "w") as f:
                    f.write(full_report_text)
                logger.debug("Reports saved to: %s", report_file)
            except Exception as e:
                logger.warning("Failed to save reports to %s: %s", report_file, e)

        return {
            "final": f_curr,
            "snapshots": snapshots,
            "times": times,
            "summary": summary,
            "history": diagnostics.history,  # Include full history for detailed plotting
            "report_text": full_report_text,
        }
