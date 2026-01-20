"""Plotting functions for diagnostic metrics and quality checks."""

import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple
from ...logging_utils import get_logger

logger = get_logger("plotting.diagnostics")


def plot_mass_conservation(
    history: Dict,
    figsize: Tuple[float, float] = (10, 5),
    title: str = "Mass Conservation",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot mass and mass change over time.

    Parameters
    ----------
    history : dict
        Diagnostics history dictionary with 'mass' and 'mass_change' keys
    figsize : tuple
        Figure size (width, height)
    title : str
        Figure title
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    plt.Figure
        The figure object
    """
    if "mass" not in history:
        logger.warning("'mass' not found in history; skipping mass conservation plot")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Mass over time
    ax1.plot(history["mass"], linewidth=2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("∫f(x)dx")
    ax1.set_title("Total Mass")
    ax1.grid(True, alpha=0.3)

    # Mass change
    ax2.plot(history["mass_change"], linewidth=2, color="C1")
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Δ(mass)")
    ax2.set_title("Mass Change per Step")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info("Saved mass conservation plot to: %s", output_path)

    return fig


def plot_entropy(
    history: Dict,
    figsize: Tuple[float, float] = (10, 5),
    title: str = "Entropy Evolution",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot entropy and entropy change over time.

    Parameters
    ----------
    history : dict
        Diagnostics history dictionary with 'entropy' and 'entropy_change' keys
    figsize : tuple
        Figure size (width, height)
    title : str
        Figure title
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    plt.Figure
        The figure object
    """
    if "entropy" not in history:
        logger.warning("'entropy' not found in history; skipping entropy plot")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Entropy
    ax1.plot(history["entropy"], linewidth=2, color="C2")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("S = -∫f ln(f) dV")
    ax1.set_title("Shannon Entropy")
    ax1.grid(True, alpha=0.3)

    # Entropy change
    ax2.plot(history["entropy_change"], linewidth=2, color="C3")
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("ΔS")
    ax2.set_title("Entropy Change per Step")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info("Saved entropy plot to: %s", output_path)

    return fig


def plot_convergence(
    history: Dict,
    figsize: Tuple[float, float] = (10, 5),
    title: str = "Solution Convergence (L2 & L∞)",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot solution norms and convergence metrics.

    Parameters
    ----------
    history : dict
        Diagnostics history with 'l2_change' and 'linf_change' keys
    figsize : tuple
        Figure size (width, height)
    title : str
        Figure title
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    plt.Figure
        The figure object
    """
    if "l2_change" not in history and "linf_change" not in history:
        logger.warning("Convergence metrics not found in history; skipping convergence plot")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Plot both norms on log scale
    if history["l2_change"]:
        ax.semilogy(history["l2_change"], linewidth=2, label="L² norm")
    if history["linf_change"]:
        ax.semilogy(history["linf_change"], linewidth=2, label="L∞ norm")

    ax.set_xlabel("Step")
    ax.set_ylabel("||Δf||")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info("Saved convergence plot to: %s", output_path)

    return fig


def plot_solution_bounds(
    history: Dict,
    figsize: Tuple[float, float] = (10, 5),
    title: str = "Solution Bounds (Min/Max)",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot min and max values of solution over time (positivity check).

    Parameters
    ----------
    history : dict
        Diagnostics history with 'min_val' and 'max_val' keys
    figsize : tuple
        Figure size (width, height)
    title : str
        Figure title
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    plt.Figure
        The figure object
    """
    if "min_val" not in history or "max_val" not in history:
        logger.warning("Solution bounds not found in history; skipping bounds plot")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(
        range(len(history["min_val"])),
        history["min_val"],
        history["max_val"],
        alpha=0.3,
        label="[min, max]",
    )
    ax.plot(history["min_val"], linewidth=2, label="min", color="C0")
    ax.plot(history["max_val"], linewidth=2, label="max", color="C1")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Step")
    ax.set_ylabel("f(x)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info("Saved solution bounds plot to: %s", output_path)

    return fig


def plot_diagnostics_summary(
    history: Dict,
    figsize: Tuple[float, float] = (14, 10),
    title: str = "Complete Diagnostics Summary",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a 2x2 grid of key diagnostic metrics.

    Parameters
    ----------
    history : dict
        Complete diagnostics history dictionary
    figsize : tuple
        Figure size (width, height)
    title : str
        Figure title
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    plt.Figure
        The figure object
    """
    if not history or not any(
        key in history for key in ["mass", "entropy", "l2_change", "min_val"]
    ):
        logger.warning("Insufficient history data for diagnostics summary; skipping")
        return None

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    # 1. Mass
    ax1.plot(history["mass"], linewidth=2, color="C0")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("∫f(x)dx")
    ax1.set_title("Mass Conservation", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # 2. Entropy
    ax2.plot(history["entropy"], linewidth=2, color="C2")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Shannon Entropy")
    ax2.set_title("Entropy Evolution", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # 3. Convergence (L2/Linf on log scale)
    if history["l2_change"]:
        ax3.semilogy(history["l2_change"], linewidth=2, label="L² norm")
    if history["linf_change"]:
        ax3.semilogy(history["linf_change"], linewidth=2, label="L∞ norm")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("||Δf||")
    ax3.set_title("Convergence (Log Scale)", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3, which="both")

    # 4. Solution bounds
    ax4.fill_between(
        range(len(history["min_val"])), history["min_val"], history["max_val"], alpha=0.3
    )
    ax4.plot(history["min_val"], linewidth=2, label="min", color="C0")
    ax4.plot(history["max_val"], linewidth=2, label="max", color="C1")
    ax4.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("f(x)")
    ax4.set_title("Solution Bounds (Positivity)", fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info("Saved diagnostics summary plot to: %s", output_path)

    return fig
