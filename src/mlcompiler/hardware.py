from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HardwareConfig:
    sram_bytes: int
    dram_bandwidth_GBs: Optional[float] = None
    compute_Gops: Optional[float] = None
