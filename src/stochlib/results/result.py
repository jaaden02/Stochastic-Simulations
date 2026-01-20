"""Core result container for stochlib solvers.

Provides unified Result class for storing and analyzing simulation outputs
from Fokker-Planck, SDE, and deterministic solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, TYPE_CHECKING
import numpy as np
from datetime import datetime

if TYPE_CHECKING:
    from ..simulation import SimulationSetup

SolverType = Literal["fokker_planck", "sde", "deterministic"]


@dataclass
class SimulationResult:
    """Unified result container for any simulation method.

    Stores solution data, metadata, and provides analysis/plotting methods.

    Attributes
    ----------
    solver_type : SolverType
        Method used ('fokker_planck', 'sde', or 'deterministic')
    data : np.ndarray
        Solution data:
        - FP: shape (nx, ny, nz, ..., nt) for probability distribution
        - SDE: shape (n_paths, nt, ndim) for particle trajectories
        - Deterministic: shape (n_trajectories, nt, ndim)
    t_array : np.ndarray
        Time points where solution was computed
    grid : Grid
        Spatial grid object used for simulation
    initial_condition : InitialCondition
        Initial condition object
    velocities : VelocitiesConfig
        Velocity/drift configuration
    diffusions : DiffusionConfig
        Diffusion configuration
    boundary_conditions : BoundaryConditions
        Boundary conditions used
    metadata : dict
        Additional info (timestamp, elapsed_time, scheme, etc.)
    """

    solver_type: SolverType
    data: np.ndarray
    t_array: np.ndarray
    grid: Any  # Grid object
    initial_condition: Any  # InitialCondition object
    velocities: Any  # VelocitiesConfig object
    diffusions: Any  # DiffusionConfig object
    boundary_conditions: Any  # BoundaryConditions object
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data consistency."""
        if len(self.data.shape) < 2:
            raise ValueError("Data must have at least 2 dimensions")
        if len(self.t_array) < 2:
            raise ValueError("t_array must have at least 2 time points")

        # Validate time dimension consistency
        if self.solver_type == "fokker_planck":
            if self.data.shape[-1] != len(self.t_array):
                raise ValueError(
                    f"FP data time dim {self.data.shape[-1]} != t_array length {len(self.t_array)}"
                )
        elif self.solver_type in ("sde", "deterministic"):
            if self.data.shape[1] != len(self.t_array):
                raise ValueError(
                    f"Path data time dim {self.data.shape[1]} != t_array length {len(self.t_array)}"
                )

        # Set default metadata
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().isoformat()
        if "solver_type" not in self.metadata:
            self.metadata["solver_type"] = self.solver_type

    @property
    def final(self) -> np.ndarray:
        """Get final state/distribution at t_final.

        Returns
        -------
        np.ndarray
            Final FP distribution (shape: spatial dims) or
            Final path endpoints (shape: n_paths, ndim)
        """
        if self.solver_type == "fokker_planck":
            return self.data[..., -1]
        else:  # SDE or deterministic
            return self.data[:, -1, :]

    @property
    def initial(self) -> np.ndarray:
        """Get initial state/distribution at t_0.

        Returns
        -------
        np.ndarray
            Initial FP distribution or initial path positions
        """
        if self.solver_type == "fokker_planck":
            return self.data[..., 0]
        else:  # SDE or deterministic
            return self.data[:, 0, :]

    @property
    def t_start(self) -> float:
        """Start time."""
        result: float = float(self.t_array[0])
        return result

    @property
    def t_final(self) -> float:
        """Final time."""
        result: float = float(self.t_array[-1])
        return result

    @property
    def duration(self) -> float:
        """Total simulated time span."""
        return self.t_final - self.t_start

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return len(self.t_array)

    def is_fokker_planck(self) -> bool:
        """Check if result is from FP solver."""
        return self.solver_type == "fokker_planck"

    def is_sde(self) -> bool:
        """Check if result is from SDE solver."""
        return self.solver_type == "sde"

    def is_deterministic(self) -> bool:
        """Check if result is from deterministic solver."""
        return self.solver_type == "deterministic"

    def statistics(self) -> Dict[str, np.ndarray]:
        """Compute ensemble statistics for SDE/deterministic paths.

        Returns
        -------
        dict
            - 'mean': ensemble mean trajectory (nt, ndim)
            - 'std': ensemble std at each time (nt, ndim)
            - 'min': ensemble minimum at each time
            - 'max': ensemble maximum at each time

        Raises
        ------
        ValueError
            If called on Fokker-Planck result (use moments() instead)
        """
        if self.solver_type == "fokker_planck":
            raise ValueError("Use moments() for FP results, not statistics()")

        # data shape: (n_paths, nt, ndim)
        return {
            "mean": np.mean(self.data, axis=0),  # (nt, ndim)
            "std": np.std(self.data, axis=0),  # (nt, ndim)
            "min": np.min(self.data, axis=0),  # (nt, ndim)
            "max": np.max(self.data, axis=0),  # (nt, ndim)
        }

    def moments(self, order: int = 2) -> Dict[str, np.ndarray]:
        """Compute spatial moments of FP distribution.

        Parameters
        ----------
        order : int, default=2
            Maximum moment order to compute (1=mean, 2=variance, etc.)

        Returns
        -------
        dict
            Keys: 'mean', 'variance', 'skewness', 'kurtosis' (depending on order)

        Raises
        ------
        ValueError
            If called on path-based result (use statistics() instead)
        NotImplementedError
            If not yet implemented for this solver type
        """
        if self.solver_type != "fokker_planck":
            raise ValueError(
                f"moments() requires Fokker-Planck distribution data, "
                f"but this result is from '{self.solver_type}' solver. "
                f"Use statistics() instead for ensemble statistics from SDE/deterministic paths. "
                f"Example: result.statistics() returns mean/std/min/max of path ensemble."
            )

        # This is a placeholder - actual implementation depends on grid dimensionality
        # and axis naming. Will be filled in during implementation.
        raise NotImplementedError(
            "moments() for FP distributions not yet implemented. "
            "Requires numerical integration over spatial grid (dimension-dependent). "
            "Workaround: Manually integrate using scipy.integrate.trapezoid(). "
            "Example: from scipy.integrate import trapezoid; "
            "mean_x = trapezoid(x_grid * result.final, dx=grid.dx). "
            "See examples/ or GitHub issues for timeline."
        )

    def warm_start(
        self, t_final_new: float, dt: Optional[float] = None, **override_params
    ) -> "SimulationSetup":
        """Prepare for continued simulation from final state.

        Creates a new SimulationSetup using the final state as initial condition
        for continuation to later time.

        Parameters
        ----------
        t_final_new : float
            New final time for continuation (must be > self.t_final)
        dt : float, optional
            Time step for new simulation. If None, uses metadata['dt'] if available.
        **override_params
            Any SimulationSetup parameters to override (velocities, diffusions, etc.)

        Returns
        -------
        SimulationSetup
            Setup ready for continued simulation from t_final to t_final_new

        Raises
        ------
        ValueError
            If t_final_new <= self.t_final
        ImportError
            If SimulationSetup not imported
        """
        if t_final_new <= self.t_final:
            raise ValueError(f"t_final_new ({t_final_new}) must be > t_final ({self.t_final})")

        # Import here to avoid circular imports
        from ..simulation import SimulationSetup

        # Create IC from final state
        if self.solver_type == "fokker_planck":
            f0_new = self.final
        else:
            raise NotImplementedError(
                f"warm_start() not yet implemented for '{self.solver_type}' results. "
                f"Reason: Re-initializing paths from final distribution requires special handling. "
                f"Workaround: Extract final positions result.data[:, -1, :], then create new "
                f"PathSimulator with these as initial positions. "
                f"See examples/warm_start_workaround.ipynb for full example. "
                f"Open GitHub issue for implementation timeline."
            )

        return SimulationSetup(
            grid=self.grid,
            initial_condition=self.initial_condition,  # Will be overridden
            velocities=override_params.get("velocities", self.velocities),
            diffusions=override_params.get("diffusions", self.diffusions),
            boundary_conditions=self.boundary_conditions,
            f0_override=f0_new,
            t_start=self.t_final,
            **override_params,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary for saving.

        Returns
        -------
        dict
            Serializable representation
        """
        return {
            "solver_type": self.solver_type,
            "data": self.data.tolist() if isinstance(self.data, np.ndarray) else self.data,
            "t_array": (
                self.t_array.tolist() if isinstance(self.t_array, np.ndarray) else self.t_array
            ),
            "metadata": self.metadata,
            # Note: grid, ic, velocities, diffusions, bc are NOT serialized
            # User must reconstruct setup and pass result.warm_start()
        }

    def save(self, filepath: str, format: str = "npy") -> None:
        """Save result to disk.

        Parameters
        ----------
        filepath : str
            Path to save file (extension ignored, format determines output)
        format : {'npy', 'npz', 'pickle'}, default='npy'
            Save format
        """
        if format == "npy":
            np.save(filepath, self.data)
            # TODO: Save metadata alongside
        elif format == "npz":
            np.savez(
                filepath,
                data=self.data,
                t_array=self.t_array,
                metadata_json=str(self.metadata),
            )
        elif format == "pickle":
            import pickle

            with open(filepath, "wb") as f:
                pickle.dump(self.to_dict(), f)
        else:
            supported = ["npy", "npz", "pickle"]
            raise ValueError(
                f"Format '{format}' not supported. Supported formats: {', '.join(supported)}. "
                f"Recommended: 'npz' for fast I/O, 'pickle' for metadata preservation."
            )

    @classmethod
    def load(cls, filepath: str, setup: "SimulationSetup", format: str = "npz") -> SimulationResult:
        """Load result from disk.

        Parameters
        ----------
        filepath : str
            Path to saved result
        setup : SimulationSetup
            Original setup (grid, ic, etc.) - result data is reattached to this
        format : {'npy', 'npz', 'pickle'}, default='npz'
            Expected file format

        Returns
        -------
        SimulationResult
            Loaded result
        """
        if format == "npz":
            loaded = np.load(filepath, allow_pickle=True)
            return cls(
                solver_type=setup.solver_type,
                data=loaded["data"],
                t_array=loaded["t_array"],
                grid=setup.grid,
                initial_condition=setup.initial_condition,
                velocities=setup.velocities,
                diffusions=setup.diffusions,
                boundary_conditions=setup.boundary_conditions,
                metadata=dict(loaded["metadata_json"]),
            )
        else:
            raise NotImplementedError(
                f"Loading from '{format}' format not yet implemented. "
                f"Currently supported: npz. Planned: HDF5 (large arrays), pickle (full metadata). "
                f"Workaround: Use numpy.load('{filepath}', allow_pickle=True) to inspect structure. "
                f"Open GitHub issue to prioritize."
            )
