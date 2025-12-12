from mlcompiler import (
    Graph,
    HardwareConfig,
    Op,
    Tensor,
    choose_schedule,
    estimate_dram_bytes,
    estimate_peak_sram_bytes,
)


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


def test_cost_model_naive_vs_memory_aware():
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
    from mlcompiler.schedule import NAIVE_SCHEDULE, MEMORY_AWARE_SCHEDULE

    dram_naive = estimate_dram_bytes(g, NAIVE_SCHEDULE).total_bytes
    dram_mem = estimate_dram_bytes(g, MEMORY_AWARE_SCHEDULE).total_bytes
    assert dram_mem < dram_naive

    peak_naive = estimate_peak_sram_bytes(g, NAIVE_SCHEDULE).peak_bytes
    peak_mem = estimate_peak_sram_bytes(g, MEMORY_AWARE_SCHEDULE).peak_bytes
    assert peak_naive <= peak_mem
