from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union


_DTYPE_BYTES: Mapping[str, int] = {
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float32": 4,
    "fp32": 4,
    "int8": 1,
    "uint8": 1,
    "int32": 4,
}


def _dtype_bytes(dtype: Union[str, int]) -> int:
    if isinstance(dtype, int):
        if dtype <= 0:
            raise ValueError(f"dtype bytes must be positive, got {dtype}")
        return dtype
    key = dtype.lower()
    if key not in _DTYPE_BYTES:
        raise ValueError(f"Unknown dtype {dtype!r}. Known: {sorted(_DTYPE_BYTES)}")
    return _DTYPE_BYTES[key]


@dataclass(frozen=True)
class Tensor:
    shape: Tuple[int, ...]
    dtype: Union[str, int] = "float16"

    def nbytes(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n * _dtype_bytes(self.dtype)


@dataclass
class Op:
    name: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, object] = field(default_factory=dict)

    def infer_output_tensors(self, input_tensors: Sequence[Tensor]) -> List[Tensor]:
        kind = self.name
        if kind == "Linear":
            if not input_tensors:
                raise ValueError("Linear op requires one input tensor")
            in_features = self.attrs.get("in_features")
            out_features = self.attrs.get("out_features")
            if in_features is None or out_features is None:
                raise ValueError("Linear op requires attrs in_features and out_features")
            in_features_i = int(in_features)
            out_features_i = int(out_features)
            inp = input_tensors[0]
            if inp.shape and inp.shape[-1] != in_features_i:
                raise ValueError(
                    f"Linear expects last dim {in_features_i}, got {inp.shape[-1]}"
                )
            out_shape = inp.shape[:-1] + (out_features_i,)
            out_dtype = self.attrs.get("dtype", inp.dtype)
            return [Tensor(out_shape, out_dtype)]

        if kind == "GELU":
            if not input_tensors:
                raise ValueError("GELU op requires one input tensor")
            inp = input_tensors[0]
            return [Tensor(inp.shape, inp.dtype)]

        raise NotImplementedError(f"Shape inference not implemented for op {kind!r}")


@dataclass
class Graph:
    ops: List[Op]
    inputs: Dict[str, Tensor]
    outputs: List[str]

    def walk_ops(self) -> Iterable[Op]:
        return iter(self.ops)

    def infer_shapes(self) -> Dict[str, Tensor]:
        tensors: Dict[str, Tensor] = dict(self.inputs)
        for op in self.ops:
            try:
                ins = [tensors[name] for name in op.inputs]
            except KeyError as e:
                missing = e.args[0]
                raise KeyError(f"Op {op.name} missing input tensor {missing!r}") from None
            outs = op.infer_output_tensors(ins)
            if len(outs) != len(op.outputs):
                raise ValueError(
                    f"Op {op.name} produced {len(outs)} outputs, "
                    f"but outputs list has {len(op.outputs)} names"
                )
            for name, tensor in zip(op.outputs, outs):
                tensors[name] = tensor
        return tensors
