"""Boundary condition management for PDE solvers.

Supports open, no-flux, and periodic boundary conditions for 1D, 2D, and 3D grids.
"""
from typing import Optional
import numpy as np

from .setup import Grid


class BoundaryConditions:
    """Boundary condition manager supporting 'open', 'noflux', and 'periodic'.

    Works with 1D, 2D, or 3D grids by specifying per-axis BCs. Provides helpers
    for periodic neighbor wrapping and boundary flux computation for advection.

    Usage
    -----
    bc = BoundaryConditions(grid, bc_x='open', bc_y='noflux', bc_z='periodic')
    # Wrap index for periodic axis
    j_next = bc.neighbor_index(j, axis='y', n_points=grid.num_points_y, offset=+1)
    # Compute boundary flux at lower/upper faces for 'open'/'noflux'
    F_lower = bc.boundary_flux(m=mu, f_cell=f[i], position='lower', axis='x')
    F_upper = bc.boundary_flux(m=mu, f_cell=f[-1], position='upper', axis='x')
    """

    OPEN = 'open'
    NOFLUX = 'noflux'
    PERIODIC = 'periodic'

    def __init__(self, grid: Grid, bc_x: str = 'open', bc_y: Optional[str] = None, bc_z: Optional[str] = None) -> None:
        """Initialize boundary conditions for present axes.

        Parameters
        ----------
        grid : Grid
            The spatial grid (used to infer present axes)
        bc_x : str
            Boundary condition for x-axis ('open' | 'noflux' | 'periodic')
        bc_y : str or None
            Boundary condition for y-axis (if present). Defaults to 'open' if axis exists.
        bc_z : str or None
            Boundary condition for z-axis (if present). Defaults to 'open' if axis exists.
        """
        self.grid: Grid = grid
        self._axis_bc: dict = {}

        # Validate and assign x BC (always present)
        self._axis_bc['x'] = self._validate_bc(bc_x)

        # y-axis if present
        if grid.y_grid is not None:
            bc_y = bc_y or self.OPEN
            self._axis_bc['y'] = self._validate_bc(bc_y)

        # z-axis if present
        if grid.z_grid is not None:
            bc_z = bc_z or self.OPEN
            self._axis_bc['z'] = self._validate_bc(bc_z)

    @staticmethod
    def _validate_bc(bc: str) -> str:
        if bc not in (BoundaryConditions.OPEN, BoundaryConditions.NOFLUX, BoundaryConditions.PERIODIC):
            raise ValueError("Boundary condition must be 'open', 'noflux', or 'periodic'")
        return bc

    def for_axis(self, axis: str) -> str:
        """Return the BC string for the given axis ('x'|'y'|'z')."""
        if axis not in self._axis_bc:
            raise ValueError(f"Axis '{axis}' not present or not configured")
        return self._axis_bc[axis]

    def is_periodic(self, axis: str) -> bool:
        """Check if the given axis uses periodic boundary conditions."""
        return self.for_axis(axis) == self.PERIODIC

    def neighbor_index(self, idx: int, axis: str, n_points: int, offset: int) -> int:
        """Get neighbor index along an axis considering the BC.

        Parameters
        ----------
        idx : int
            Current index
        axis : str
            Axis label ('x'|'y'|'z')
        n_points : int
            Number of points along the axis
        offset : int
            Neighbor offset (+1 for right/top/next, -1 for left/bottom/prev)

        Returns
        -------
        int
            Neighbor index with periodic wrap if applicable; otherwise clamped.
        """
        bc = self.for_axis(axis)
        j = idx + offset
        if bc == self.PERIODIC:
            # Pythonic wrap
            return (j + n_points) % n_points
        # Clamp for open/noflux
        if j < 0:
            return 0
        if j >= n_points:
            return n_points - 1
        return j

    def boundary_flux(self, m: float, f_cell: float, position: str, axis: str) -> Optional[float]:
        """Compute boundary face flux for advection at domain edges.

        Implements:
        - 'open': allow outflow, block inflow
          lower face: flux = m*f if m < 0 else 0
          upper face: flux = m*f if m > 0 else 0
        - 'noflux': flux = 0
        - 'periodic': boundary flux not used (neighbors wrap); returns None

        Parameters
        ----------
        m : float
            Drift/velocity at the boundary cell
        f_cell : float
            Cell value at the boundary
        position : str
            'lower' or 'upper' face of the axis
        axis : str
            Axis label ('x'|'y'|'z')

        Returns
        -------
        float or None
            Flux value; None for periodic since solver should use wrapped neighbors.
        """
        bc = self.for_axis(axis)
        if bc == self.NOFLUX:
            return 0.0
        if bc == self.PERIODIC:
            return None
        # OPEN
        if position == 'lower':
            return m * f_cell if m < 0.0 else 0.0
        if position == 'upper':
            return m * f_cell if m > 0.0 else 0.0
        raise ValueError("position must be 'lower' or 'upper'")
    