from typing import Optional, Dict, Any, List, Tuple, Callable
import numpy as np


class Grid:
    """Defines a flexible spatial domain that supports 1D, 2D, or 3D."""
    
    def __init__(self,
                 x_start: float, x_end: float, num_points_x: int,
                 y_start: Optional[float] = None, y_end: Optional[float] = None, num_points_y: Optional[int] = None,
                 z_start: Optional[float] = None, z_end: Optional[float] = None, num_points_z: Optional[int] = None) -> None:
        """
        Initialize a spatial grid in 1D, 2D, or 3D depending on provided axes.
        
        Parameters
        ----------
        x_start, x_end : float
            Spatial domain boundaries in x-direction
        num_points_x : int
            Number of grid points in x-direction
        y_start, y_end : float, float, optional
            Spatial domain boundaries in y-direction
        num_points_y : int, optional
            Number of grid points in y-direction
        z_start, z_end : float, float, optional
            Spatial domain boundaries in z-direction
        num_points_z : int, optional
            Number of grid points in z-direction
        """
        # Store domain boundaries (x always present)
        self.x_start, self.x_end = x_start, x_end
        self.num_points_x = num_points_x
        self.x_grid = np.linspace(x_start, x_end, num_points_x)
        self.dx = self.x_grid[1] - self.x_grid[0] if num_points_x and num_points_x > 1 else 0.0

        # Optional y-axis
        self.y_start, self.y_end, self.num_points_y = y_start, y_end, num_points_y
        self.y_grid = None
        self.dy = 0.0
        if y_start is not None and y_end is not None and num_points_y is not None and num_points_y > 0:
            self.y_grid = np.linspace(y_start, y_end, num_points_y)
            self.dy = self.y_grid[1] - self.y_grid[0] if num_points_y > 1 else 0.0

        # Optional z-axis
        self.z_start, self.z_end, self.num_points_z = z_start, z_end, num_points_z
        self.z_grid = None
        self.dz = 0.0
        if z_start is not None and z_end is not None and num_points_z is not None and num_points_z > 0:
            self.z_grid = np.linspace(z_start, z_end, num_points_z)
            self.dz = self.z_grid[1] - self.z_grid[0] if num_points_z > 1 else 0.0

        # Build axis list in canonical order
        self.axis_names = ['x']
        grids = [self.x_grid]
        if self.y_grid is not None:
            self.axis_names.append('y')
            grids.append(self.y_grid)
        if self.z_grid is not None:
            self.axis_names.append('z')
            grids.append(self.z_grid)

        # Create meshgrids for present axes
        mesh = np.meshgrid(*grids, indexing='ij')
        # Map meshes back to attributes and dict for dynamic access
        self.mesh = {}
        # Always set X
        self.X = mesh[0]
        self.mesh['x'] = mesh[0]
        # Conditionally set Y and Z
        if len(mesh) > 1:
            self.Y = mesh[1]
            self.mesh['y'] = mesh[1]
        else:
            self.Y = None
        if len(mesh) > 2:
            self.Z = mesh[2]
            self.mesh['z'] = mesh[2]
        else:
            self.Z = None
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the spatial grid in present dimensions."""
        dims = [self.num_points_x]
        if self.y_grid is not None:
            dims.append(self.num_points_y)
        if self.z_grid is not None:
            dims.append(self.num_points_z)
        return tuple(dims)
    
    @property
    def deltas(self) -> Dict[str, float]:
        """Return grid spacings per axis as a dictionary."""
        result: Dict[str, float] = {'x': self.dx}
        if self.y_grid is not None:
            result['y'] = self.dy
        if self.z_grid is not None:
            result['z'] = self.dz
        return result
    
    @property
    def total_points(self) -> int:
        """Return the total number of grid points (product of all dimensions)."""
        return int(np.prod(self.shape))
    
    @property
    def volume_element(self) -> float:
        """Return the volume element as the product of present spacings."""
        vol = self.dx
        if self.y_grid is not None:
            vol *= self.dy
        if self.z_grid is not None:
            vol *= self.dz
        return vol
    
    def __repr__(self) -> str:
        dims_repr: List[str] = [f"x=[{self.x_start}, {self.x_end}]"]
        if self.y_grid is not None:
            dims_repr.append(f"y=[{self.y_start}, {self.y_end}]")
        if self.z_grid is not None:
            dims_repr.append(f"z=[{self.z_start}, {self.z_end}]")
        return (f"Grid({', '.join(dims_repr)}, shape={self.shape})")

class InitialCondition:
    """Handles initial condition setup for the simulation."""
    
    def __init__(self, grid: Grid, func_type: str = "gaussian", normalize: bool = True, **params: Any) -> None:
        """
        Initialize an initial condition on a given grid.
        
        Parameters
        ----------
        grid : Grid
            The spatial grid to evaluate the initial condition on
        func_type : str
            Type of initial condition ('gaussian', 'uniform', 'delta', etc.)
        normalize : bool
            Whether to normalize the distribution to integrate to 1
        **params : dict
            Parameters specific to the chosen function type
        """
        self.grid = grid
        self.func_type = func_type
        self.normalize = normalize
        self.params = params
        
        # Set default parameters if not provided
        self._set_default_params()
        
        # Compute the initial condition
        self.f0 = self._compute()
        
        # Normalize if requested
        if self.normalize:
            self._normalize()
    
    def _set_default_params(self) -> None:
        """Set sensible defaults for missing parameters based on present axes."""
        # Always set x defaults
        self.params.setdefault('x0', (self.grid.x_start + self.grid.x_end) / 2)
        self.params.setdefault('sigma_x', 2 * self.grid.dx)
        # Conditionally set y/z defaults
        if self.grid.y_grid is not None:
            self.params.setdefault('y0', (self.grid.y_start + self.grid.y_end) / 2)
            self.params.setdefault('sigma_y', 2 * self.grid.dy)
        if self.grid.z_grid is not None:
            self.params.setdefault('z0', (self.grid.z_start + self.grid.z_end) / 2)
            self.params.setdefault('sigma_z', 2 * self.grid.dz)
    
    def _compute(self) -> np.ndarray:
        """Compute the initial condition field for 1D, 2D, or 3D."""
        # TODO: write docstring explaining supported types
        if self.func_type == "gaussian":
            return self._gaussian_nd()
        elif self.func_type == "uniform":
            return np.ones(self.grid.shape)
        elif self.func_type == "delta":
            return self._delta_nd()
        elif self.func_type == "custom":
            # Allow user to pass a custom function; call with present mesh arrays
            if 'func' not in self.params:
                raise ValueError("Custom initial condition requires 'func' parameter")
            meshes = [self.grid.mesh[name] for name in self.grid.axis_names]
            return self.params['func'](*meshes)
        else:
            raise ValueError(f"Unknown initial condition type: {self.func_type}")
    
    def _gaussian_nd(self) -> np.ndarray:
        """Compute N-D Gaussian initial condition over present axes."""
        total_arg = (self.grid.X - self.params['x0'])**2 / (2 * self.params['sigma_x']**2)
        if self.grid.y_grid is not None:
            total_arg += (self.grid.Y - self.params['y0'])**2 / (2 * self.params['sigma_y']**2)
        if self.grid.z_grid is not None:
            total_arg += (self.grid.Z - self.params['z0'])**2 / (2 * self.params['sigma_z']**2)
        return np.exp(-total_arg)
    
    def _delta_nd(self) -> np.ndarray:
        """Compute delta-like initial condition concentrated at a point in N-D."""
        # Indices per axis
        i = np.argmin(np.abs(self.grid.x_grid - self.params['x0']))
        idx = [i]
        if self.grid.y_grid is not None:
            j = np.argmin(np.abs(self.grid.y_grid - self.params['y0']))
            idx.append(j)
        if self.grid.z_grid is not None:
            k = np.argmin(np.abs(self.grid.z_grid - self.params['z0']))
            idx.append(k)
        
        f0 = np.zeros(self.grid.shape)
        f0[tuple(idx)] = 1.0
        return f0
    
    def _normalize(self) -> None:
        """Normalize the distribution to integrate to 1."""
        integral = np.sum(self.f0) * self.grid.volume_element
        if integral > 0:
            self.f0 = self.f0 / integral
    
    def __repr__(self) -> str:
        return (f"InitialCondition(type='{self.func_type}', "
                f"normalized={self.normalize}, "
                f"integral={np.sum(self.f0) * self.grid.volume_element:.6f})")


class DiffusionConfig:
    """Configure diffusion per axis (x/y/z) with constants or functions.

    Supports 1D/2D/3D grids. You can specify:
    - axes where diffusion is active (subset of present axes)
    - constant coefficients per axis (e.g., D_x = 0.01)
    - functions per axis returning a field over the grid (may be time-dependent)

    Examples
    --------
    # 1) Constant diffusion only in x
    cfg = DiffusionConfig(grid, axes=['x'], constants={'x': 0.01})

    # 2) Spatially varying diffusion in y, periodic in x with constant
    cfg = DiffusionConfig(
        grid,
        axes=['x','y'],
        constants={'x': 0.01},
        functions={'y': lambda X, Y, *args: 0.02 + 0.01*np.sin(Y)}
    )

    # 3) Time-dependent diffusion in z
    cfg = DiffusionConfig(
        grid,
        axes=['z'],
        functions={'z': lambda X, Y, Z, t: 0.005 + 0.002*np.cos(t)}
    )
    D_fields = cfg.build_fields(t=0.5)  # {'z': array(...)}
    """

    # TODO: support for anisotropic diffusion tensors could be added later
    # TODO: support for cross-axis diffusion terms could be added later
    # TODO: distinguish between scalar and tensor diffusion later if needed
    # TODO: 
    def __init__(self,
                 grid: Grid,
                 axes: Optional[List[str]] = None,
                 constants: Optional[Dict[str, float]] = None,
                 functions: Optional[Dict[str, Callable]] = None) -> None:
        self.grid = grid
        # Default: diffuse along all present axes
        self.axes = axes[:] if axes is not None else grid.axis_names[:]
        self.constants = constants.copy() if constants is not None else {}
        self.functions = functions.copy() if functions is not None else {}

        # Validate axes exist on grid
        present = set(grid.axis_names)
        for a in self.axes:
            if a not in present:
                raise ValueError(f"Axis '{a}' not present on grid; present={grid.axis_names}")

        # Ensure provided specs refer only to declared axes
        for a in list(self.constants.keys()):
            if a not in self.axes:
                raise ValueError(f"Constant provided for axis '{a}' not in active axes {self.axes}")
        for a in list(self.functions.keys()):
            if a not in self.axes:
                raise ValueError(f"Function provided for axis '{a}' not in active axes {self.axes}")

    def build_fields(self, t: Optional[float] = None, include_inactive: bool = False, evaluate: bool = True) -> Dict[str, Any]:
        """Return per-axis diffusion definitions or evaluated fields.

        If evaluate=True (default): returns ndarray fields matching grid shape.
        If evaluate=False: constants stay scalar; functions are returned as callables.

        Function signature when evaluated:
        - 1D: func(X) or func(X, t)
        - 2D: func(X, Y) or func(X, Y, t)
        - 3D: func(X, Y, Z) or func(X, Y, Z, t)
        """
        meshes = [self.grid.mesh[a] for a in self.grid.axis_names]
        D_fields = {}
        for a in self.axes:
            func = self.functions.get(a)
            const = self.constants.get(a)

            if not evaluate:
                if func is not None:
                    D_fields[a] = func  # caller may evaluate lazily
                elif const is not None:
                    D_fields[a] = float(const)
                else:
                    raise ValueError(f"Axis '{a}' requires either a constant or function for diffusion")
                continue

            # evaluate=True path
            if func is not None:
                if t is None:
                    D_fields[a] = func(*meshes)
                else:
                    D_fields[a] = func(*meshes, t)
            elif const is not None:
                D_fields[a] = np.full(self.grid.shape, float(const), dtype=float)
            else:
                raise ValueError(f"Axis '{a}' requires either a constant or function for diffusion")

            # Basic shape validation
            if D_fields[a].shape != self.grid.shape:
                raise ValueError(
                    f"Diffusion field for axis '{a}' has shape {D_fields[a].shape}, expected {self.grid.shape}"
                )
            if np.any(D_fields[a] < 0):
                raise ValueError(f"Diffusion field for axis '{a}' contains negative values")

        if include_inactive:
            for a in self.grid.axis_names:
                if a not in D_fields:
                    if evaluate:
                        D_fields[a] = np.zeros(self.grid.shape, dtype=float)
                    else:
                        D_fields[a] = 0.0

        return D_fields

import numpy as np

class VelocitiesConfig:
    """
    Configure advection velocities per axis (x/y/z) with constants or functions.
    Supports 1D, 2D, or 3D grids dynamically.
    """
    def __init__(self, grid: Grid, mu_x: Optional[Any] = None, mu_y: Optional[Any] = None, mu_z: Optional[Any] = None) -> None:
        self.grid: Grid = grid
        # Store provided definitions in a mapping for easier iteration
        self._specs: Dict[str, Optional[Any]] = {'x': mu_x, 'y': mu_y, 'z': mu_z}
        
        # Validate that provided velocities correspond to present axes
        present_axes: set = set(grid.axis_names)
        for axis, val in self._specs.items():
            if val is not None and axis not in present_axes:
                raise ValueError(f"mu_{axis} provided but {axis}-axis not present on grid")

    def build_fields(self, t: Optional[float] = None, evaluate: bool = True) -> Dict[str, np.ndarray]:
        """
        Return per-axis velocity definitions or evaluated fields matching grid shape.
        
        If evaluate=True (default): returns ndarray fields (useful for Fokker-Planck kernels).
        If evaluate=False: returns the raw constant or callable.
        """
        # Get only the mesh arrays relevant to the current dimensionality (e.g., [X, Y] for 2D)
        meshes: List[np.ndarray] = [self.grid.mesh[a] for a in self.grid.axis_names]
        mu_fields: Dict[str, np.ndarray] = {}
        
        for axis in self.grid.axis_names:
            mu = self._specs.get(axis)
            if mu is None:
                # Default to zero velocity if axis exists but no mu provided
                if evaluate:
                    mu_fields[axis] = np.zeros(self.grid.shape, dtype=float)
                continue
                
            if not evaluate:
                mu_fields[axis] = mu
                continue
            
            # evaluate=True path: handle functions vs constants
            if callable(mu):
                # Use *meshes to pass the correct number of spatial arguments dynamically
                try:
                    field = mu(*meshes, t)
                except TypeError as e:
                    # If time-dependent call fails, try without time argument
                    try:
                        field = mu(*meshes)
                    except TypeError:
                        raise ValueError(
                            f"Velocity function for axis '{axis}' has incorrect signature. "
                            f"Expected one of: func(*meshes) or func(*meshes, t). "
                            f"Got error: {e}"
                        )
            else:
                field = np.full(self.grid.shape, float(mu), dtype=float)
            
            # Ensure the output matches grid dimensions
            if field.shape != self.grid.shape:
                field = np.broadcast_to(field, self.grid.shape)
            
            mu_fields[axis] = field.astype(float)
        
        return mu_fields

    def max_velocity_magnitude(self, t: Optional[float] = None) -> float:
        """
        Compute the maximum absolute velocity magnitude over the grid.
        Essential for CFL stability checks.
        """
        mu_fields = self.build_fields(t=t, evaluate=True)
        if not mu_fields:
            return 0.0
        # Computes max(|mu|) for each axis and returns the global maximum
        return max(np.max(np.abs(field)) for field in mu_fields.values())