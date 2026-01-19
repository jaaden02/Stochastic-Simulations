"""Low-level numerical kernels for Fokker-Planck PDE stepping.

Provides JIT-compiled kernels for efficient finite difference schemes including
Chang-Cooper, upwind, and Lax-Wendroff methods.
"""

import numpy as np
from numba import njit, prange
from typing import Tuple

# Internal Numba Flags
BC_OPEN: int = 0
BC_NOFLUX: int = 1
BC_PERIODIC: int = 2

SCHEME_CHANG_COOPER: int = 0
SCHEME_CENTRAL_CN: int = 1
SCHEME_UPWIND_CN: int = 2


@njit(inline="always")
def _delta_cc_scalar(Pe: float) -> float:
    """Stability-optimized Chang-Cooper weighting."""
    if abs(Pe) < 1e-4:
        return 0.5 - Pe / 12.0
    if Pe > 100.0:
        return 1.0 / Pe
    if Pe < -100.0:
        return 1.0 / Pe + 1.0
    return 1.0 / Pe - 1.0 / (np.exp(Pe) - 1.0)


@njit(inline="always")
def calculate_flux(
    f_curr: float,
    f_next: float,
    mu_f: float,
    D_f: float,
    d_coord: float,
    inv_d: float,
    scheme: int,
) -> float:
    """Calculates face flux based on the chosen numerical scheme."""
    if scheme == SCHEME_CHANG_COOPER:
        Pe = mu_f * d_coord / (D_f + 1e-16)
        delta = _delta_cc_scalar(Pe)
        return mu_f * ((1.0 - delta) * f_curr + delta * f_next) - D_f * (f_next - f_curr) * inv_d

    elif scheme == SCHEME_CENTRAL_CN:
        return mu_f * 0.5 * (f_curr + f_next) - D_f * (f_next - f_curr) * inv_d

    else:  # SCHEME_UPWIND_CN
        val_adv = f_curr if mu_f >= 0 else f_next
        return mu_f * val_adv - D_f * (f_next - f_curr) * inv_d


@njit(parallel=True, fastmath=True)
def universal_fp_step_kernel(
    f: np.ndarray,
    MU: Tuple[np.ndarray, np.ndarray, np.ndarray],
    D: Tuple[np.ndarray, np.ndarray, np.ndarray],
    d_coords: Tuple[float, float, float],
    dt: float,
    bc_flags: np.ndarray,
    scheme_flags: np.ndarray,
) -> np.ndarray:
    """
    3D Fokker-Planck Kernel supporting mixed schemes and BCs per axis.

    Parameters
    ----------
    f : ndarray, shape (Nx, Ny, Nz)
        Probability/distribution field
    MU : tuple of ndarray
        Velocity fields (MUX, MUY, MUZ) per axis
    D : tuple of ndarray
        Diffusion fields (DX, DY, DZ) per axis
    d_coords : tuple of float
        Grid spacings (dx, dy, dz)
    dt : float
        Time step
    bc_flags : ndarray, shape (3,)
        Boundary condition flags for (x, y, z)
    scheme_flags : ndarray, shape (3,)
        Scheme selection flags for (x, y, z)

    Returns
    -------
    ndarray
        Updated field after one time step
    """
    Nx, Ny, Nz = f.shape
    f_new = np.zeros_like(f)

    dx, dy, dz = d_coords
    idx, idy, idz = 1.0 / dx, 1.0 / dy, 1.0 / dz

    # Unpack fields for clarity
    MUX, MUY, MUZ = MU
    DX, DY, DZ = D

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):

                # --- AXIS X ---
                # Left Face
                if i == 0:
                    if bc_flags[0] == BC_PERIODIC:
                        F_x_l = calculate_flux(
                            f[Nx - 1, j, k],
                            f[i, j, k],
                            MUX[i, j, k],
                            DX[i, j, k],
                            dx,
                            idx,
                            scheme_flags[0],
                        )
                    else:
                        F_x_l = (
                            MUX[i, j, k] * f[i, j, k]
                            if (bc_flags[0] == BC_OPEN and MUX[i, j, k] < 0)
                            else 0.0
                        )
                else:
                    F_x_l = calculate_flux(
                        f[i - 1, j, k],
                        f[i, j, k],
                        0.5 * (MUX[i - 1, j, k] + MUX[i, j, k]),
                        0.5 * (DX[i - 1, j, k] + DX[i, j, k]),
                        dx,
                        idx,
                        scheme_flags[0],
                    )

                # Right Face
                if i == Nx - 1:
                    if bc_flags[0] == BC_PERIODIC:
                        F_x_r = calculate_flux(
                            f[i, j, k],
                            f[0, j, k],
                            MUX[i, j, k],
                            DX[i, j, k],
                            dx,
                            idx,
                            scheme_flags[0],
                        )
                    else:
                        F_x_r = (
                            MUX[i, j, k] * f[i, j, k]
                            if (bc_flags[0] == BC_OPEN and MUX[i, j, k] > 0)
                            else 0.0
                        )
                else:
                    F_x_r = calculate_flux(
                        f[i, j, k],
                        f[i + 1, j, k],
                        0.5 * (MUX[i, j, k] + MUX[i + 1, j, k]),
                        0.5 * (DX[i, j, k] + DX[i + 1, j, k]),
                        dx,
                        idx,
                        scheme_flags[0],
                    )

                # --- AXIS Y ---
                if j == 0:
                    if bc_flags[1] == BC_PERIODIC:
                        F_y_l = calculate_flux(
                            f[i, Ny - 1, k],
                            f[i, j, k],
                            MUY[i, j, k],
                            DY[i, j, k],
                            dy,
                            idy,
                            scheme_flags[1],
                        )
                    else:
                        F_y_l = (
                            MUY[i, j, k] * f[i, j, k]
                            if (bc_flags[1] == BC_OPEN and MUY[i, j, k] < 0)
                            else 0.0
                        )
                else:
                    F_y_l = calculate_flux(
                        f[i, j - 1, k],
                        f[i, j, k],
                        0.5 * (MUY[i, j - 1, k] + MUY[i, j, k]),
                        0.5 * (DY[i, j - 1, k] + DY[i, j, k]),
                        dy,
                        idy,
                        scheme_flags[1],
                    )

                if j == Ny - 1:
                    if bc_flags[1] == BC_PERIODIC:
                        F_y_r = calculate_flux(
                            f[i, j, k],
                            f[i, 0, k],
                            MUY[i, j, k],
                            DY[i, j, k],
                            dy,
                            idy,
                            scheme_flags[1],
                        )
                    else:
                        F_y_r = (
                            MUY[i, j, k] * f[i, j, k]
                            if (bc_flags[1] == BC_OPEN and MUY[i, j, k] > 0)
                            else 0.0
                        )
                else:
                    F_y_r = calculate_flux(
                        f[i, j, k],
                        f[i, j + 1, k],
                        0.5 * (MUY[i, j, k] + MUY[i, j + 1, k]),
                        0.5 * (DY[i, j, k] + DY[i, j + 1, k]),
                        dy,
                        idy,
                        scheme_flags[1],
                    )

                # --- AXIS Z ---
                if k == 0:
                    if bc_flags[2] == BC_PERIODIC:
                        F_z_l = calculate_flux(
                            f[i, j, Nz - 1],
                            f[i, j, k],
                            MUZ[i, j, k],
                            DZ[i, j, k],
                            dz,
                            idz,
                            scheme_flags[2],
                        )
                    else:
                        F_z_l = (
                            MUZ[i, j, k] * f[i, j, k]
                            if (bc_flags[2] == BC_OPEN and MUZ[i, j, k] < 0)
                            else 0.0
                        )
                else:
                    F_z_l = calculate_flux(
                        f[i, j, k - 1],
                        f[i, j, k],
                        0.5 * (MUZ[i, j, k - 1] + MUZ[i, j, k]),
                        0.5 * (DZ[i, j, k - 1] + DZ[i, j, k]),
                        dz,
                        idz,
                        scheme_flags[2],
                    )

                if k == Nz - 1:
                    if bc_flags[2] == BC_PERIODIC:
                        F_z_r = calculate_flux(
                            f[i, j, k],
                            f[i, j, 0],
                            MUZ[i, j, k],
                            DZ[i, j, k],
                            dz,
                            idz,
                            scheme_flags[2],
                        )
                    else:
                        F_z_r = (
                            MUZ[i, j, k] * f[i, j, k]
                            if (bc_flags[2] == BC_OPEN and MUZ[i, j, k] > 0)
                            else 0.0
                        )
                else:
                    F_z_r = calculate_flux(
                        f[i, j, k],
                        f[i, j, k + 1],
                        0.5 * (MUZ[i, j, k] + MUZ[i, j, k + 1]),
                        0.5 * (DZ[i, j, k] + DZ[i, j, k + 1]),
                        dz,
                        idz,
                        scheme_flags[2],
                    )

                # Divergence update
                div = -((F_x_r - F_x_l) * idx + (F_y_r - F_y_l) * idy + (F_z_r - F_z_l) * idz)
                f_new[i, j, k] = max(0.0, f[i, j, k] + dt * div)

    return f_new
