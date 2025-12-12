from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class OpKind(str, Enum):
    LINEAR = "Linear"
    GELU = "GELU"
    ADD = "Add"
    MATMUL = "MatMul"


@dataclass(frozen=True)
class TensorType:
    shape: Tuple[int, ...]
    dtype_bytes: int = 2

    @property
    def nbytes(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n * self.dtype_bytes


@dataclass
class Node:
    name: str
    op: OpKind
    inputs: List[str]
    output: TensorType
    attrs: Dict[str, object] = field(default_factory=dict)


@dataclass
class Graph:
    nodes: List[Node]
    outputs: List[str]
    inputs: List[str] = field(default_factory=list)

    def node_by_name(self, name: str) -> Optional[Node]:
        for n in self.nodes:
            if n.name == name:
                return n
        return None

    def topo_order(self) -> List[Node]:
        # For now assume nodes already topologically sorted.
        return list(self.nodes)

