from __future__ import annotations

from dataclasses import dataclass
from .hardware import HardwareConfig
from .ir import Graph
from .schedule import MEMORY_AWARE_SCHEDULE, NAIVE_SCHEDULE, Schedule
from .cost import evaluate_schedule


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
    naive_cost = evaluate_schedule(graph, NAIVE_SCHEDULE, hw)
    mem_cost = evaluate_schedule(graph, MEMORY_AWARE_SCHEDULE, hw)
    best = min([naive_cost, mem_cost], key=lambda c: c.penalized_cost)
    fits = mem_cost.feasible
    schedule = best.schedule
    return ScheduleChoice(
        schedule=schedule,
        fits_in_sram=fits,
        estimated_intermediate_bytes=inter_bytes,
    )
