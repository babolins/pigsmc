from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import eval_legendre


class FreeRotorGrid:
    """Precomputed free-rotor propagator G(x; lambda_rot) on a 1-D grid.

    G(x) = Σ_{l=0}^{L_max} (2l+1)/(4π) P_l(x) exp(-lambda_rot * l*(l+1))

    Uses Fritsch-Carlson (PCHIP) monotone cubic interpolation so that
    interpolated values are guaranteed non-negative.
    """

    def __init__(
        self,
        lambda_rot: float,
        L_max: int = 100,
        grid_size: int = 1000,
    ) -> None:
        self._lambda_rot = lambda_rot
        self._L_max = L_max
        x = np.linspace(-1.0, 1.0, grid_size)
        G = np.zeros_like(x)
        for l in range(L_max + 1):
            weight = (2 * l + 1) / (4.0 * np.pi) * np.exp(-lambda_rot * l * (l + 1))
            G += weight * eval_legendre(l, x)
        # Clamp tiny negatives from truncation before taking log
        G = np.maximum(G, 0.0)
        # PCHIP interpolator on G for eval(); on log(G+eps) for log_eval()
        self._pchip_G = PchipInterpolator(x, G)
        log_G = np.log(np.maximum(G, 1e-300))
        self._pchip_logG = PchipInterpolator(x, log_G)

    def eval(self, x: float | np.ndarray) -> float | np.ndarray:
        """Return G(x) >= 0, x clamped to [-1, 1]."""
        x = np.clip(x, -1.0, 1.0)
        return np.maximum(self._pchip_G(x), 0.0)

    def log_eval(self, x: float | np.ndarray) -> float | np.ndarray:
        """Return log G(x), x clamped to [-1, 1]."""
        x = np.clip(x, -1.0, 1.0)
        return self._pchip_logG(x)
