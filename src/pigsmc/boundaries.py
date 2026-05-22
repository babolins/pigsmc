from __future__ import annotations

import numpy as np


class Boundary:
    """Base class for boundary conditions. Use factory methods to construct."""

    @staticmethod
    def free() -> FreeBoundary:
        return FreeBoundary()

    @staticmethod
    def periodic_3d(Lx: float, Ly: float, Lz: float) -> Periodic3dBoundary:
        return Periodic3dBoundary(Lx, Ly, Lz)

    @staticmethod
    def quasi_2d(Lx: float, Ly: float) -> Quasi2dBoundary:
        return Quasi2dBoundary(Lx, Ly)

    @staticmethod
    def quasi_1d(Lz: float) -> Quasi1dBoundary:
        return Quasi1dBoundary(Lz)

    def displacement(self, r_i, r_j) -> np.ndarray:
        raise NotImplementedError

    def all_displacements(self, positions_slice: np.ndarray) -> np.ndarray:
        r = np.asarray(positions_slice, dtype=float)  # (N, 3)
        return self.displacement(r[:, np.newaxis, :], r[np.newaxis, :, :])

    @property
    def _kind(self) -> int:
        raise NotImplementedError

    @property
    def _box_half(self) -> np.ndarray:
        raise NotImplementedError


class FreeBoundary(Boundary):

    @property
    def _kind(self) -> int:
        return 0

    @property
    def _box_half(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)

    def displacement(self, r_i, r_j) -> np.ndarray:
        return np.asarray(r_i, dtype=float) - np.asarray(r_j, dtype=float)


class Periodic3dBoundary(Boundary):

    def __init__(self, Lx: float, Ly: float, Lz: float):
        for name, L in (("Lx", Lx), ("Ly", Ly), ("Lz", Lz)):
            if L <= 0:
                raise ValueError(
                    f"All box lengths must be positive; got {name}={L}"
                )
        self._Lx = float(Lx)
        self._Ly = float(Ly)
        self._Lz = float(Lz)

    @property
    def box_lengths(self) -> np.ndarray:
        return np.array([self._Lx, self._Ly, self._Lz], dtype=np.float64)

    @property
    def _kind(self) -> int:
        return 1

    @property
    def _box_half(self) -> np.ndarray:
        return np.array(
            [self._Lx / 2, self._Ly / 2, self._Lz / 2], dtype=np.float64
        )

    def displacement(self, r_i, r_j) -> np.ndarray:
        d = np.asarray(r_i, dtype=float) - np.asarray(r_j, dtype=float)
        L = self.box_lengths
        return d - L * np.round(d / L)


class Quasi2dBoundary(Boundary):
    """Periodic in x and y; free in z."""

    def __init__(self, Lx: float, Ly: float):
        for name, L in (("Lx", Lx), ("Ly", Ly)):
            if L <= 0:
                raise ValueError(
                    f"All box lengths must be positive; got {name}={L}"
                )
        self._Lx = float(Lx)
        self._Ly = float(Ly)

    @property
    def box_lengths(self) -> np.ndarray:
        return np.array([self._Lx, self._Ly], dtype=np.float64)

    @property
    def _kind(self) -> int:
        return 2

    @property
    def _box_half(self) -> np.ndarray:
        return np.array([self._Lx / 2, self._Ly / 2, 0.0], dtype=np.float64)

    def displacement(self, r_i, r_j) -> np.ndarray:
        d = np.asarray(r_i, dtype=float) - np.asarray(r_j, dtype=float)
        d[..., 0] = d[..., 0] - self._Lx * np.round(d[..., 0] / self._Lx)
        d[..., 1] = d[..., 1] - self._Ly * np.round(d[..., 1] / self._Ly)
        return d


class Quasi1dBoundary(Boundary):
    """Free in x and y; periodic in z."""

    def __init__(self, Lz: float):
        if Lz <= 0:
            raise ValueError(f"All box lengths must be positive; got Lz={Lz}")
        self._Lz = float(Lz)

    @property
    def box_lengths(self) -> np.ndarray:
        return np.array([self._Lz], dtype=np.float64)

    @property
    def _kind(self) -> int:
        return 3

    @property
    def _box_half(self) -> np.ndarray:
        return np.array([0.0, 0.0, self._Lz / 2], dtype=np.float64)

    def displacement(self, r_i, r_j) -> np.ndarray:
        d = np.asarray(r_i, dtype=float) - np.asarray(r_j, dtype=float)
        d[..., 2] = d[..., 2] - self._Lz * np.round(d[..., 2] / self._Lz)
        return d
