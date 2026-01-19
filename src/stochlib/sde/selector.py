"""Heuristics for choosing SDE step schemes."""

from typing import Callable, Optional
import numpy as np
from ..logging_utils import get_logger

logger = get_logger("sde.selector")
Array = np.ndarray


class StepSchemeAdvisor:
    """Simple heuristics to pick Euler-Maruyama vs Milstein."""

    @staticmethod
    def choose_scheme(
        scheme: str,
        diffusion: Callable[[Array, float], Array],
        diffusion_jacobian: Optional[Callable[[Array, float], Array]] = None,
    ) -> str:
        if scheme == "auto":
            if diffusion_jacobian is not None:
                return "milstein"
            # Probe state dependence: compare sigma at two points
            try:
                sigma0 = diffusion(np.zeros((1, 1)), 0.0)
                sigma1 = diffusion(np.ones((1, 1)), 0.0)
                if np.any(np.abs(sigma0 - sigma1) > 1e-12):
                    return "milstein"
            except Exception:
                pass
            return "euler_maruyama"
        scheme = scheme.lower()
        if scheme not in {"euler_maruyama", "milstein"}:
            raise ValueError(f"Unknown scheme '{scheme}'")
        return scheme
