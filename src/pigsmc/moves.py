from __future__ import annotations

from enum import IntEnum

from pigsmc._engine import (
    Move,
    MoveResult,
    PathState,
    RotationBisectionMove,
    RotationEndMove,
    RotationInteriorMove,
    RotationRigidMove,
    TranslationBisectionMove,
    TranslationEndMove,
    TranslationInteriorMove,
    TranslationRigidMove,
)


class SliceKind(IntEnum):
    PHYSICAL = 0
    INTERMEDIATE = 1


__all__ = [
    "Move",
    "MoveResult",
    "PathState",
    "RotationBisectionMove",
    "RotationEndMove",
    "RotationInteriorMove",
    "RotationRigidMove",
    "SliceKind",
    "TranslationBisectionMove",
    "TranslationEndMove",
    "TranslationInteriorMove",
    "TranslationRigidMove",
]
