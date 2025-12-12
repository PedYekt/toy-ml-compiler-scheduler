from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class ScheduleKind(str, Enum):
    NAIVE = "naive"
    FUSED = "fused"


@dataclass(frozen=True)
class Schedule:
    kind: ScheduleKind
    order: List[str]

