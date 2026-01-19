"""Fokker-Planck equation solver implementation.

Provides the core FokkerPlanckSolver class for numerically solving Fokker-Planck PDEs
using various finite difference schemes.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from .kernel import universal_fp_step_kernel
from ..boundary_conditions import BoundaryConditions
from ..setup import Grid
from ..logging_utils import get_logger

logger = get_logger("fokker_planck.solver")


class FokkerPlanckSolver:
    def __init__(
        self,
        bc_manager: BoundaryConditions,
        schemes: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the Fokker-Planck solver.

        Parameters
        ----------
        bc_manager : BoundaryConditions
            Boundary condition configuration
        schemes : dict, optional
            Scheme selection per axis; keys are 'x', 'y', 'z'.
            Defaults to Chang-Cooper on all axes.
        """
        if schemes is None:
            schemes = {"x": "chang_cooper", "y": "chang_cooper", "z": "chang_cooper"}

        self.bc_manager: BoundaryConditions = bc_manager
        self.grid: Grid = bc_manager.grid
        self.schemes: Dict[str, str] = schemes
        self.axis_names: List[str] = self.grid.axis_names

        # Build BC flags only for axes that exist on the grid
        bc_map: Dict[str, int] = {"open": 0, "noflux": 1, "periodic": 2}
        bc_flags_list: List[int] = []
        for ax in self.axis_names:
            bc_flags_list.append(bc_map[bc_manager.for_axis(ax)])
        # Pad to 3 for Numba kernel compatibility
        while len(bc_flags_list) < 3:
            bc_flags_list.append(1)  # default to noflux for inactive axes
        self.bc_flags: np.ndarray = np.array(bc_flags_list, dtype=np.int32)

        # Build scheme flags only for axes that exist on the grid
        scheme_map: Dict[str, int] = {"chang_cooper": 0, "central_cn": 1, "upwind_cn": 2}
        scheme_flags_list: List[int] = []
        for ax in self.axis_names:
            scheme_flags_list.append(scheme_map[schemes.get(ax, "chang_cooper")])
        # Pad to 3 for Numba kernel compatibility
        while len(scheme_flags_list) < 3:
            scheme_flags_list.append(0)  # default to chang_cooper for inactive axes
        self.scheme_flags: np.ndarray = np.array(scheme_flags_list, dtype=np.int32)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Solver init: axes=%s bc_flags=%s scheme_flags=%s grid_shape=%s",
                self.axis_names,
                self.bc_flags,
                self.scheme_flags,
                self.grid.shape,
            )

    def solve_step(
        self,
        f: np.ndarray,
        MU_fields: Dict[str, np.ndarray],
        D_fields: Dict[str, np.ndarray],
        dt: float,
    ) -> np.ndarray:
        """Advance one time step using the Fokker-Planck kernel.

        Parameters
        ----------
        f : ndarray
            Current probability/distribution field (shape matches grid.shape)
        MU_fields : dict
            Velocity fields per axis; keys correspond to grid.axis_names
        D_fields : dict
            Diffusion fields per axis; keys correspond to grid.axis_names
        dt : float
            Time step

        Returns
        -------
        ndarray
            Updated field after one step

        Raises
        ------
        ValueError
            If field shapes don't match grid dimensions or if required fields are missing
        """
        # Validate field shape
        if f.shape != self.grid.shape:
            raise ValueError(f"Field shape {f.shape} does not match grid shape {self.grid.shape}")

        # Validate that all active axes have velocity and diffusion fields
        for ax in self.axis_names:
            if ax not in MU_fields:
                raise ValueError(f"Missing velocity field for axis '{ax}'")
            if ax not in D_fields:
                raise ValueError(f"Missing diffusion field for axis '{ax}'")

            # Check field shapes
            if MU_fields[ax].shape != self.grid.shape:
                raise ValueError(
                    f"Velocity field for axis '{ax}' has shape {MU_fields[ax].shape}, "
                    f"expected {self.grid.shape}"
                )
            if D_fields[ax].shape != self.grid.shape:
                raise ValueError(
                    f"Diffusion field for axis '{ax}' has shape {D_fields[ax].shape}, "
                    f"expected {self.grid.shape}"
                )

        # Normalize shape to 3D for kernel (pad singleton dims for 1D/2D)
        original_shape = f.shape
        if f.ndim == 1:
            f_work = f[:, None, None]
        elif f.ndim == 2:
            f_work = f[:, :, None]
        else:
            f_work = f
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "solve_step: original_shape=%s work_shape=%s dt=%.3e",
                original_shape,
                f_work.shape,
                dt,
            )

        # Package fields as tuples for the kernel, padding with zeros for missing axes.
        # Broadcast provided fields to match padded work shape.
        MU_list: List[np.ndarray] = []
        D_list: List[np.ndarray] = []

        for ax in ["x", "y", "z"]:
            if ax in self.axis_names:
                mu_arr = MU_fields[ax]
                d_arr = D_fields[ax]
                # Pad trailing singleton dims until 3D, then broadcast to work shape
                while mu_arr.ndim < 3:
                    mu_arr = mu_arr[..., None]
                while d_arr.ndim < 3:
                    d_arr = d_arr[..., None]
                MU_list.append(np.broadcast_to(mu_arr, f_work.shape))
                D_list.append(np.broadcast_to(d_arr, f_work.shape))
            else:
                # Inactive axis: use zero arrays matching work shape
                MU_list.append(np.zeros_like(f_work))
                D_list.append(np.zeros_like(f_work))

        MU: Tuple[np.ndarray, np.ndarray, np.ndarray] = tuple(MU_list)  # type: ignore
        D: Tuple[np.ndarray, np.ndarray, np.ndarray] = tuple(D_list)  # type: ignore

        # Assemble grid spacings for all three dimensions
        d_coords: Tuple[float, float, float] = (
            self.grid.dx,
            self.grid.dy if self.grid.y_grid is not None else 1.0,
            self.grid.dz if self.grid.z_grid is not None else 1.0,
        )

        f_new = universal_fp_step_kernel(
            f_work, MU, D, d_coords, dt, self.bc_flags, self.scheme_flags
        )

        # Squeeze back to original dimensionality
        if original_shape == f_new.shape:
            return f_new
        if len(original_shape) == 1:
            return f_new[:, 0, 0]
        if len(original_shape) == 2:
            return f_new[:, :, 0]
        return f_new
