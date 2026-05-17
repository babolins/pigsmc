from __future__ import annotations

from enum import IntEnum

from pigsmc._engine import (
    Move,
    MoveResult,
    PathState,
    TranslationEndMove,
    TranslationInteriorMove,
)


class SliceKind(IntEnum):
    PHYSICAL = 0
    INTERMEDIATE = 1


__all__ = [
    "Move",
    "MoveResult",
    "PathState",
    "SliceKind",
    "TranslationEndMove",
    "TranslationInteriorMove",
]
