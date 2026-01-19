"""Diagnostic tools for Fokker-Planck solutions.

Provides stability analysis, solution quality metrics, and diagnostic reporting
for Fokker-Planck PDE solutions.
"""

import numpy as np
import logging
from typing import Optional, Dict
from ..setup import Grid
from ..boundary_conditions import BoundaryConditions
from ..logging_utils import get_logger

logger = get_logger("fokker_planck.diagnostics")


class DiagnosticReport:
    """Object returned by the analyzer containing limits and warnings."""

    def __init__(self, dt_max: float, memory_gb: float, is_stable: bool):
        self.dt_max: float = dt_max
        self.memory_gb: float = memory_gb
        self.is_stable: bool = is_stable


class StabilityAnalyzer:
    @staticmethod
    def analyze(
        grid: Grid,
        velocities,
        diffusions,
        dt_user: float,
        t: Optional[float] = 0.0,
    ) -> DiagnosticReport:
        """Analyze stability and resource requirements for a given configuration.

        Parameters
        ----------
        grid : Grid
            Spatial grid configuration
        velocities : VelocitiesConfig
            Advection velocity configuration
        diffusions : DiffusionConfig
            Diffusion coefficient configuration
        dt_user : float
            Requested time step
        t : float, optional
            Current time (for time-dependent coefficients)

        Returns
        -------
        DiagnosticReport
            Report with max stable time step, memory usage, and stability flag
        """
        # Calculate rates using your existing notebook logic
        max_mu: float = velocities.max_velocity_magnitude(t=t)

        # Advection and Diffusion Rates
        deltas: Dict[str, float] = grid.deltas
        adv_rates: float = sum([max_mu / deltas[ax] for ax in grid.axis_names])

        # D_phi = D0 logic from your notebook
        diff_fields: Dict = diffusions.build_fields(t=t, evaluate=True)
        diff_rates: float = sum(
            [(2.0 * np.max(diff_fields[ax])) / (deltas[ax] ** 2) for ax in grid.axis_names]
        )

        dt_max: float = 1.0 / (adv_rates + diff_rates + 1e-16)

        # Resource check
        mem: float = (grid.total_points * 8) / 1e9  # GB for one 64-bit float array

        is_stable: bool = dt_user <= dt_max

        return DiagnosticReport(dt_max, mem, is_stable)


# ============================================================================
# RUNTIME DIAGNOSTICS (Monitor solution during simulation)
# ============================================================================


class SolutionDiagnostics:
    """Runtime checks: mass conservation, positivity, entropy, convergence."""

    def __init__(
        self,
        grid: Grid,
        volume_element: float,
        boundary_conditions: Optional[BoundaryConditions] = None,
        report_interval: int = 1,
        mass_tolerance: float = 1e-3,
        on_report: Optional[callable] = None,
    ):
        """Initialize runtime diagnostics tracker.

        Parameters
        ----------
        grid : Grid
            Spatial grid configuration
        volume_element : float
            Grid volume element (dx * dy * dz)
        boundary_conditions : BoundaryConditions, optional
            If provided, mass loss is permitted when any axis is 'open'.
        report_interval : int
            Report every N steps (>=1). Set to 0 to disable reports. Default: 1 (every step).
        mass_tolerance : float
            Threshold for flagging mass change.
        """
        self.grid: Grid = grid
        self.volume_element: float = volume_element
        self.boundary_conditions: Optional[BoundaryConditions] = boundary_conditions
        self.report_interval: int = int(report_interval) if report_interval > 0 else 0
        self.mass_tolerance: float = mass_tolerance
        self.step_count: int = 0
        self._on_report = on_report
        self._reports_disabled: bool = report_interval == 0

        # Mass loss allowed if any configured axis is open; otherwise closed system.
        self._mass_loss_allowed: bool = False
        if self.boundary_conditions is not None:
            self._mass_loss_allowed = any(
                self.boundary_conditions.for_axis(ax) == BoundaryConditions.OPEN
                for ax in grid.axis_names
            )

        # History tracking
        self.history: Dict = {
            "mass": [],
            "mass_change": [],
            "min_val": [],
            "max_val": [],
            "has_negative": [],
            "entropy": [],
            "entropy_change": [],
            "l2_change": [],
            "linf_change": [],
        }
        self.f_prev: Optional[np.ndarray] = None
        self.warnings: list = []

    def analyze_step(self, f: np.ndarray, t: float) -> Dict:
        """Analyze solution at current time step.

        Parameters
        ----------
        f : ndarray
            Current solution field
        t : float
            Current time

        Returns
        -------
        dict
            Metrics for this step with keys: mass, min_val, max_val, has_negative,
            entropy, l2_change, linf_change, warnings, should_report
        """
        self.step_count += 1
        should_report: bool = (not self._reports_disabled) and (
            self.step_count % max(1, self.report_interval)
        ) == 0

        metrics: Dict = {"should_report": should_report}
        step_warnings: list = []

        # 1. Mass Conservation
        mass: float = np.sum(f) * self.volume_element
        metrics["mass"] = mass
        metrics["mass_change"] = mass - self.history["mass"][-1] if self.history["mass"] else 0.0
        self.history["mass"].append(mass)
        self.history["mass_change"].append(metrics["mass_change"])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("t=%.3e mass=%.6e dmass=%.3e", t, mass, metrics["mass_change"])

        # BC-aware mass warning logic
        if self._mass_loss_allowed:
            if metrics["mass_change"] > self.mass_tolerance:
                step_warnings.append(
                    f"t={t:.4e}: Mass increased by {metrics['mass_change']:.2e} with open BC"
                )
        else:
            if abs(metrics["mass_change"]) > self.mass_tolerance:
                step_warnings.append(
                    f"t={t:.4e}: Mass change {metrics['mass_change']:.2e} in closed BC"
                )

        # 2. Positivity Check
        min_val: float = np.min(f)
        max_val: float = np.max(f)
        metrics["min_val"] = min_val
        metrics["max_val"] = max_val
        self.history["min_val"].append(min_val)
        self.history["max_val"].append(max_val)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("t=%.3e min=%.3e max=%.3e", t, min_val, max_val)

        has_negative: bool = min_val < -1e-10
        metrics["has_negative"] = has_negative
        self.history["has_negative"].append(has_negative)

        if has_negative:
            step_warnings.append(f"t={t:.4e}: Negative values detected (min={min_val:.2e})")

        # 3. Entropy (Shannon-like): S = -∫ f ln(f) dV
        f_safe: np.ndarray = np.where(f > 1e-16, f, 1e-16)
        entropy: float = -np.sum(f_safe * np.log(f_safe)) * self.volume_element
        metrics["entropy"] = entropy
        self.history["entropy"].append(entropy)

        entropy_change: float = (
            entropy - self.history["entropy"][-2] if len(self.history["entropy"]) > 1 else 0.0
        )
        metrics["entropy_change"] = entropy_change
        self.history["entropy_change"].append(entropy_change)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("t=%.3e entropy=%.6e dentropy=%.3e", t, entropy, entropy_change)

        if entropy_change < -1e-4:
            step_warnings.append(f"t={t:.4e}: Entropy decreased (ΔS={entropy_change:.2e})")

        # 4. Convergence: L2 and L∞ norms of change
        if self.f_prev is not None:
            df: np.ndarray = f - self.f_prev
            l2_change: float = np.sqrt(np.sum(df**2) * self.volume_element)
            linf_change: float = np.max(np.abs(df))
            metrics["l2_change"] = l2_change
            metrics["linf_change"] = linf_change
            self.history["l2_change"].append(l2_change)
            self.history["linf_change"].append(linf_change)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("t=%.3e l2=%.3e linf=%.3e", t, l2_change, linf_change)
        else:
            metrics["l2_change"] = np.inf
            metrics["linf_change"] = np.inf

        self.f_prev = f.copy()

        # 5. Compile warnings (store all, return only if report step)
        self.warnings.extend(step_warnings)
        metrics["warnings"] = step_warnings if should_report else []

        # Emit user-facing feedback when a report is due
        if should_report:
            if self._on_report is not None:
                self._on_report(step_warnings, metrics)
            else:
                header = "======== RUNTIME REPORT ========"
                footer = "================================"
                lines = [
                    header,
                    f"step       : {self.step_count}",
                    f"time       : {t:.4e}",
                    f"mass       : {metrics['mass']:.6e} (Δ={metrics['mass_change']:.2e})",
                    f"min / max  : {metrics['min_val']:.3e} / {metrics['max_val']:.3e}",
                    f"entropy    : {metrics['entropy']:.6e} (Δ={metrics['entropy_change']:.2e})",
                    f"L2 / Linf  : {metrics['l2_change']:.3e} / {metrics['linf_change']:.3e}",
                    f"negatives  : {'yes' if metrics['has_negative'] else 'no'}",
                ]
                if step_warnings:
                    lines.append("warnings  :" + ("" if len(step_warnings) == 1 else ""))
                    for w in step_warnings:
                        lines.append(f"  - {w}")
                lines.append(f"report interval: every {self.report_interval} step(s)")
                lines.append(footer)
                print("\n" + "\n".join(lines))

        return metrics

    def summary(self) -> Dict:
        """Return summary of diagnostics over all recorded steps.

        Returns
        -------
        dict
            Summary including means, totals, and critical flags
        """
        if not self.history["mass"]:
            return {"error": "No steps recorded"}

        summary: Dict = {
            "num_steps": len(self.history["mass"]),
            "total_mass_change": abs(self.history["mass"][-1] - self.history["mass"][0]),
            "mean_mass": np.mean(self.history["mass"]),
            "min_solution_value": np.min(self.history["min_val"]),
            "max_solution_value": np.max(self.history["max_val"]),
            "any_negative": any(self.history["has_negative"]),
            "entropy_start": self.history["entropy"][0] if self.history["entropy"] else 0.0,
            "entropy_end": self.history["entropy"][-1] if self.history["entropy"] else 0.0,
            "entropy_is_monotonic": (
                all(
                    self.history["entropy_change"][i] >= -1e-6
                    for i in range(1, len(self.history["entropy_change"]))
                )
                if len(self.history["entropy_change"]) > 1
                else True
            ),
            "total_warnings": len(self.warnings),
            "critical_warnings": [w for w in self.warnings if "negative" in w or "Entropy" in w],
        }

        return summary
