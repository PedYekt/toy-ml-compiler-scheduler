from __future__ import annotations

from dataclasses import dataclass
from .hardware import HardwareConfig
from .ir import Graph
from .schedule import Schedule
from .pass_schedule import run_schedule_pass


@dataclass(frozen=True)
class ScheduleChoice:
    schedule: Schedule
    fits_in_sram: bool
    estimated_intermediate_bytes: int


def estimate_intermediate_bytes(graph: Graph) -> int:
    tensors = graph.infer_shapes()
    total = 0
    for op in graph.walk_ops():
        for out_name in op.outputs:
            if out_name in graph.outputs:
                continue
            total += tensors[out_name].nbytes()
    return total


def choose_schedule(graph: Graph, hw: HardwareConfig) -> ScheduleChoice:
    inter_bytes = estimate_intermediate_bytes(graph)
    result = run_schedule_pass(graph, hw)
    mem_cost = result.costs.get("memory_aware")
    fits = mem_cost.feasible if mem_cost is not None else inter_bytes <= hw.sram_bytes
    schedule = result.chosen_schedule
    return ScheduleChoice(
        schedule=schedule,
        fits_in_sram=fits,
        estimated_intermediate_bytes=inter_bytes,
    )
