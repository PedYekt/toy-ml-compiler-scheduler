from __future__ import annotations

from dataclasses import dataclass
from .hardware import HardwareConfig
from .ir import Graph
from .schedule import Schedule, ScheduleKind


@dataclass(frozen=True)
class ScheduleChoice:
    schedule: Schedule
    fits_in_sram: bool
    estimated_intermediate_bytes: int


def estimate_intermediate_bytes(graph: Graph) -> int:
    total = 0
    for node in graph.topo_order():
        if node.name in graph.outputs:
            continue
        total += node.output.nbytes
    return total


def choose_schedule(graph: Graph, hw: HardwareConfig) -> ScheduleChoice:
    inter_bytes = estimate_intermediate_bytes(graph)
    fits = inter_bytes <= hw.sram_bytes
    if fits:
        order = [n.name for n in graph.topo_order()]
        schedule = Schedule(kind=ScheduleKind.FUSED, order=order)
    else:
        order = [n.name for n in graph.topo_order()]
        schedule = Schedule(kind=ScheduleKind.NAIVE, order=order)
    return ScheduleChoice(
        schedule=schedule,
        fits_in_sram=fits,
        estimated_intermediate_bytes=inter_bytes,
    )
