from mlcompiler import Graph, HardwareConfig, Node, OpKind, TensorType, choose_schedule


def test_choose_schedule_runs():
    g = Graph(
        nodes=[
            Node("linear1", OpKind.LINEAR, ["x"], TensorType((2, 4))),
            Node("gelu", OpKind.GELU, ["linear1"], TensorType((2, 4))),
            Node("linear2", OpKind.LINEAR, ["gelu"], TensorType((2, 2))),
        ],
        outputs=["linear2"],
        inputs=["x"],
    )
    hw = HardwareConfig(name="test", sram_bytes=1024)
    choice = choose_schedule(g, hw)
    assert choice.schedule.kind in {"fused", "naive"}

