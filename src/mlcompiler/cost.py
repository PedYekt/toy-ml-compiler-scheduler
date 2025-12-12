from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .hardware import HardwareConfig
from .ir import Graph, Tensor
from .schedule import MEMORY_AWARE_SCHEDULE, NAIVE_SCHEDULE, Schedule


def _prod(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _effective_bytes(tensor: Tensor, tile_shape: Optional[Tuple[int, ...]]) -> int:
    full_bytes = tensor.nbytes()
    if tile_shape is None:
        return full_bytes
    full_elems = _prod(tensor.shape)
    if full_elems == 0:
        return 0
    bytes_per_elem = full_bytes // full_elems
    tile_elems = _prod(tile_shape)
    tile_bytes = tile_elems * bytes_per_elem
    return min(full_bytes, tile_bytes)


@dataclass(frozen=True)
class DramEstimate:
    total_bytes: int
    breakdown: Dict[str, int]

    def __str__(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in self.breakdown.items())
        return f"DRAM(total={self.total_bytes}, {parts})"


@dataclass(frozen=True)
class SramEstimate:
    peak_bytes: int
    breakdown: Dict[str, int]

    def __str__(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in self.breakdown.items())
        return f"SRAM(peak={self.peak_bytes}, {parts})"


@dataclass(frozen=True)
class ScheduleCost:
    schedule: Schedule
    dram: DramEstimate
    sram: SramEstimate
    feasible: bool
    penalized_cost: int

    def __str__(self) -> str:
        feas = "feasible" if self.feasible else "infeasible"
        return f"{self.schedule.name}: {feas}, dram={self.dram.total_bytes}, peak_sram={self.sram.peak_bytes}"


def estimate_dram_bytes(graph: Graph, schedule: Schedule) -> DramEstimate:
    tensors = graph.infer_shapes()

    input_read = sum(t.nbytes() for t in graph.inputs.values())
    output_write = sum(tensors[name].nbytes() for name in graph.outputs)

    inter_bytes = 0
    if schedule.name == "naive":
        for op in graph.walk_ops():
            for out_name in op.outputs:
                if out_name in graph.outputs:
                    continue
                inter_bytes += tensors[out_name].nbytes()
        inter_write = inter_bytes
        inter_read = inter_bytes
        total = input_read + output_write + inter_write + inter_read
        breakdown = {
            "input_read": input_read,
            "intermediate_write": inter_write,
            "intermediate_read": inter_read,
            "output_write": output_write,
        }
        return DramEstimate(total_bytes=total, breakdown=breakdown)

    if schedule.name == "memory_aware":
        total = input_read + output_write
        breakdown = {
            "input_read": input_read,
            "intermediate_write": 0,
            "intermediate_read": 0,
            "output_write": output_write,
        }
        return DramEstimate(total_bytes=total, breakdown=breakdown)

    raise ValueError(f"Unknown schedule {schedule.name!r}")


def estimate_peak_sram_bytes(graph: Graph, schedule: Schedule) -> SramEstimate:
    tensors = graph.infer_shapes()

    if schedule.name == "naive":
        all_bytes = [t.nbytes() for t in graph.inputs.values()]
        for t in tensors.values():
            all_bytes.append(t.nbytes())
        peak = max(all_bytes) if all_bytes else 0
        return SramEstimate(peak_bytes=peak, breakdown={"peak_single_tensor": peak})

    if schedule.name == "memory_aware":
        inter_resident = 0
        for op in graph.walk_ops():
            for out_name in op.outputs:
                if out_name in graph.outputs:
                    continue
                inter_resident += _effective_bytes(
                    tensors[out_name], schedule.tile_shape
                )
        breakdown = {"intermediate_resident": inter_resident}
        if schedule.tile_shape is not None:
            breakdown["tile_shape_elems"] = _prod(schedule.tile_shape)
        return SramEstimate(peak_bytes=inter_resident, breakdown=breakdown)

    raise ValueError(f"Unknown schedule {schedule.name!r}")


def evaluate_schedule(
    graph: Graph,
    schedule: Schedule,
    hw: HardwareConfig,
    infeasible_penalty: int = 10**15,
) -> ScheduleCost:
    dram = estimate_dram_bytes(graph, schedule)
    sram = estimate_peak_sram_bytes(graph, schedule)
    feasible = sram.peak_bytes <= hw.sram_bytes
    penalty = 0 if feasible else infeasible_penalty
    penalized = dram.total_bytes + penalty
    return ScheduleCost(
        schedule=schedule,
        dram=dram,
        sram=sram,
        feasible=feasible,
        penalized_cost=penalized,
    )


def evaluate_candidates(graph: Graph, hw: HardwareConfig) -> Dict[str, ScheduleCost]:
    return {
        "naive": evaluate_schedule(graph, NAIVE_SCHEDULE, hw),
        "memory_aware": evaluate_schedule(graph, MEMORY_AWARE_SCHEDULE, hw),
    }

