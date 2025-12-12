from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


TileShape = Tuple[int, ...]


@dataclass(frozen=True)
class Schedule:
    name: str
    description: str
    tile_shape: Optional[TileShape] = None


NAIVE_SCHEDULE = Schedule(
    name="naive",
    description="Materialize each op output to DRAM before the next op.",
)

MEMORY_AWARE_SCHEDULE = Schedule(
    name="memory_aware",
    description="Keep intermediates on-chip; conceptually tile if needed.",
    tile_shape=None,
)
