from mlcompiler import Graph, HardwareConfig, Op, Tensor, choose_schedule


def test_choose_schedule_runs():
    g = Graph(
        ops=[
            Op(
                name="Linear",
                inputs=["x"],
                outputs=["linear1"],
                attrs={"in_features": 4, "out_features": 4},
            ),
            Op(name="GELU", inputs=["linear1"], outputs=["gelu"], attrs={}),
            Op(
                name="Linear",
                inputs=["gelu"],
                outputs=["linear2"],
                attrs={"in_features": 4, "out_features": 2},
            ),
        ],
        inputs={"x": Tensor((2, 4))},
        outputs=["linear2"],
    )
    tensors = g.infer_shapes()
    assert tensors["linear2"].shape == (2, 2)
    hw = HardwareConfig(sram_bytes=1024)
    choice = choose_schedule(g, hw)
    assert choice.schedule.name in {"memory_aware", "naive"}
