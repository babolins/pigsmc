from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class SliceKind(IntEnum):
    PHYSICAL = 0
    INTERMEDIATE = 1


@dataclass
class MoveResult:
    changed: list[tuple[int, int]]  # (particle_idx, slice_idx)
    log_ratio_contrib: float


class PathState:
    def __init__(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        buffer_positions: np.ndarray,
        buffer_orientations: np.ndarray,
        N: int,
        M: int,
        tau_prime: float,
        lambda_trans: np.ndarray,
        slice_kind: np.ndarray,
    ):
        pos_view = positions.view()
        pos_view.flags.writeable = False
        self._positions = pos_view

        ori_view = orientations.view()
        ori_view.flags.writeable = False
        self._orientations = ori_view

        self._buffer_positions = buffer_positions
        self._buffer_orientations = buffer_orientations
        self._N = N
        self._M = M
        self._tau_prime = tau_prime
        self._lambda_trans = lambda_trans
        self._slice_kind = slice_kind

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def orientations(self) -> np.ndarray:
        return self._orientations

    @property
    def buffer_positions(self) -> np.ndarray:
        return self._buffer_positions

    @property
    def buffer_orientations(self) -> np.ndarray:
        return self._buffer_orientations

    @property
    def N(self) -> int:
        return self._N

    @property
    def M(self) -> int:
        return self._M

    @property
    def tau_prime(self) -> float:
        return self._tau_prime

    @property
    def lambda_trans(self) -> np.ndarray:
        return self._lambda_trans

    @property
    def slice_kind(self) -> np.ndarray:
        return self._slice_kind


class Move:
    def propose(self, path_state: PathState, rng: np.random.Generator) -> MoveResult:
        raise NotImplementedError


class TranslationEndMove(Move):
    def __init__(self, step_size: float):
        self.step_size = step_size

    def propose(self, path_state: PathState, rng: np.random.Generator) -> MoveResult:
        i = int(rng.integers(0, path_state.N))
        m = int(rng.choice([0, path_state.M - 1]))
        displacement = rng.uniform(-self.step_size, self.step_size, size=3)
        path_state.buffer_positions[m, 0, :] = path_state.positions[m, i, :] + displacement
        return MoveResult(changed=[(i, m)], log_ratio_contrib=0.0)


class TranslationInteriorMove(Move):
    def __init__(self, step_size: float):
        self.step_size = step_size

    def propose(self, path_state: PathState, rng: np.random.Generator) -> MoveResult:
        i = int(rng.integers(0, path_state.N))
        m = int(rng.integers(1, path_state.M - 1))
        displacement = rng.uniform(-self.step_size, self.step_size, size=3)
        path_state.buffer_positions[m, 0, :] = path_state.positions[m, i, :] + displacement
        return MoveResult(changed=[(i, m)], log_ratio_contrib=0.0)
