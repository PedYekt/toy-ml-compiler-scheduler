from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .cost import ScheduleCost, evaluate_schedule
from .hardware import HardwareConfig
from .ir import Graph
from .schedule import MEMORY_AWARE_SCHEDULE, NAIVE_SCHEDULE, Schedule


def generate_candidates(_graph: Graph) -> List[Schedule]:
    # Phase 3 defines two valid schedules.
    return [NAIVE_SCHEDULE, MEMORY_AWARE_SCHEDULE]


def _format_bytes(nbytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(nbytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}B"
            if value.is_integer():
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{nbytes}B"


@dataclass(frozen=True)
class CompilationResult:
    chosen_schedule: Schedule
    costs: Dict[str, ScheduleCost]
    reason: str


def run_schedule_pass(
    graph: Graph,
    hw: HardwareConfig,
    candidates: Optional[Iterable[Schedule]] = None,
) -> CompilationResult:
    if candidates is None:
        candidates = generate_candidates(graph)

    costs: Dict[str, ScheduleCost] = {}
    for schedule in candidates:
        costs[schedule.name] = evaluate_schedule(graph, schedule, hw)

    feasible_costs = [c for c in costs.values() if c.feasible]
    if feasible_costs:
        best = min(
            feasible_costs,
            key=lambda c: (
                c.dram.total_bytes,
                0 if c.schedule.name == "memory_aware" else 1,
            ),
        )
    else:
        best = min(costs.values(), key=lambda c: c.penalized_cost)

    reason = "chosen by cost model"
    naive = costs.get("naive")
    mem = costs.get("memory_aware")

    if feasible_costs:
        if best.schedule.name == "memory_aware":
            if naive and naive.feasible:
                reason = (
                    f"memory_aware chosen: dram {_format_bytes(best.dram.total_bytes)} "
                    f"< naive {_format_bytes(naive.dram.total_bytes)}"
                )
            elif naive and not naive.feasible:
                reason = (
                    f"naive infeasible: peak SRAM {_format_bytes(naive.sram.peak_bytes)} "
                    f"> {_format_bytes(hw.sram_bytes)}"
                )
            else:
                reason = "memory_aware chosen: lowest DRAM among feasible schedules"
        else:
            if mem and not mem.feasible:
                reason = (
                    f"memory_aware infeasible: peak SRAM {_format_bytes(mem.sram.peak_bytes)} "
                    f"> {_format_bytes(hw.sram_bytes)}"
                )
            elif mem and mem.feasible:
                reason = (
                    f"naive chosen: dram {_format_bytes(best.dram.total_bytes)} "
                    f"<= memory_aware {_format_bytes(mem.dram.total_bytes)}"
                )
            else:
                reason = "naive chosen: lowest DRAM among feasible schedules"
    else:
        infeasible_msgs = []
        for c in costs.values():
            infeasible_msgs.append(
                f"{c.schedule.name} peak SRAM {_format_bytes(c.sram.peak_bytes)} "
                f"> {_format_bytes(hw.sram_bytes)}"
            )
        reason = (
            f"all schedules infeasible; chose {best.schedule.name} with lowest penalized cost: "
            + "; ".join(infeasible_msgs)
        )

    return CompilationResult(
        chosen_schedule=best.schedule,
        costs=costs,
        reason=reason,
    )

