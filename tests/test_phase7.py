from mlcompiler import (
    Graph,
    HardwareConfig,
    Op,
    Tensor,
    run_schedule_pass,
    estimate_dram_bytes,
    estimate_peak_sram_bytes,
)


def _make_linear_gelu_linear(batch: int, hidden: int, ff: int) -> Graph:
    ops = [
        Op(
            name="Linear",
            inputs=["x"],
            outputs=["linear1"],
            attrs={"in_features": hidden, "out_features": ff},
        ),
        Op(name="GELU", inputs=["linear1"], outputs=["gelu"], attrs={}),
        Op(
            name="Linear",
            inputs=["gelu"],
            outputs=["linear2"],
            attrs={"in_features": ff, "out_features": hidden},
        ),
    ]
    return Graph(ops=ops, inputs={"x": Tensor((batch, hidden))}, outputs=["linear2"])


def test_schedule_changes_with_sram():
    g = _make_linear_gelu_linear(batch=32, hidden=4096, ff=16384)
    hw_small = HardwareConfig(sram_bytes=1 * 1024 * 1024)  # 1MB
    hw_large = HardwareConfig(sram_bytes=8 * 1024 * 1024)  # 8MB

    small_res = run_schedule_pass(g, hw_small)
    large_res = run_schedule_pass(g, hw_large)

    assert small_res.chosen_schedule.name == "naive"
    assert large_res.chosen_schedule.name == "memory_aware"


def test_cost_monotonicity():
    g = _make_linear_gelu_linear(batch=32, hidden=1024, ff=4096)
    hw = HardwareConfig(sram_bytes=1 * 1024 * 1024)

    from mlcompiler.schedule import NAIVE_SCHEDULE, MEMORY_AWARE_SCHEDULE

    dram_naive = estimate_dram_bytes(g, NAIVE_SCHEDULE).total_bytes
    dram_mem = estimate_dram_bytes(g, MEMORY_AWARE_SCHEDULE).total_bytes
    peak_mem = estimate_peak_sram_bytes(g, MEMORY_AWARE_SCHEDULE).peak_bytes

    assert peak_mem <= hw.sram_bytes
    assert dram_mem <= dram_naive


def test_shape_inference():
    g = _make_linear_gelu_linear(batch=2, hidden=8, ff=16)
    tensors = g.infer_shapes()

    assert tensors["linear1"].shape == (2, 16)
    assert tensors["gelu"].shape == (2, 16)
    assert tensors["linear2"].shape == (2, 8)

