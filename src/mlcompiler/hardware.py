from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareConfig:
    name: str
    sram_bytes: int
    dram_bandwidth_gbps: float = 200.0

