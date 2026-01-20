"""Diagnostic visualization for deterministic PDE solutions."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from ..diagnostic import DeterministicDiagnostics


def plot_diagnostics(
    diagnostics: "DeterministicDiagnostics",
    axes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Plot diagnostic time series.

    Parameters
    ----------
    diagnostics : DeterministicDiagnostics
        Diagnostics object with history
    axes : ndarray of Axes, optional
        Array of axes (2x2). If None, creates new figure.

    Returns
    -------
    axes : ndarray of Axes
    """
    history = diagnostics.get_history()
    t = history["time"]

    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Deterministic Solution Diagnostics")

    # Mass conservation
    axes[0, 0].plot(t, history["mass"], "b-", linewidth=2)
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Mass")
    axes[0, 0].set_title("Mass Conservation")
    axes[0, 0].grid(True, alpha=0.3)

    # Mean position
    axes[0, 1].plot(t, history["mean"], "r-", linewidth=2)
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Mean Position")
    axes[0, 1].set_title("Mean Advection")
    axes[0, 1].grid(True, alpha=0.3)

    # L2 norm and total variation
    ax1 = axes[1, 0]
    ax2 = ax1.twinx()

    l1 = ax1.plot(t, history["l2_norm"], "g-", linewidth=2, label="L2 Norm")
    l2 = ax2.plot(t, history["total_variation"], "m--", linewidth=2, label="Total Variation")

    ax1.set_xlabel("Time")
    ax1.set_ylabel("L2 Norm", color="g")
    ax2.set_ylabel("Total Variation", color="m")
    ax1.set_title("Solution Norms")
    ax1.grid(True, alpha=0.3)

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    # Min/max values
    axes[1, 1].plot(t, history["max_value"], "r-", linewidth=2, label="Max")
    axes[1, 1].plot(t, history["min_value"], "b-", linewidth=2, label="Min")
    axes[1, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].set_title("Solution Range")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    return axes
