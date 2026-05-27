from __future__ import annotations

import numpy as np

from pigsmc._engine import FreeRotorGridC as _FreeRotorGridC


class FreeRotorGrid:
    """Python wrapper around the C++ FreeRotorGridC propagator.

    Provides numpy-compatible log_eval for use in tests and analysis;
    the hot path in the simulation engine calls the C++ object directly.
    """

    def __init__(
        self,
        lambda_rot: float,
        L_max: int = 100,
        grid_size: int = 1000,
    ) -> None:
        self._c = _FreeRotorGridC(lambda_rot, L_max=L_max, grid_size=grid_size)

    def log_eval(self, x: float | np.ndarray) -> float | np.ndarray:
        """Return log G(x), x clamped to [-1, 1]."""
        clipped = np.clip(x, -1.0, 1.0)
        return np.vectorize(self._c.log_eval)(clipped)
