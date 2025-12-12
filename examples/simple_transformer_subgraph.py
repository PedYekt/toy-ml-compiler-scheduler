from mlcompiler import Graph, HardwareConfig, Op, Tensor, run_schedule_pass


def make_linear_gelu_linear(batch: int, hidden: int, ff: int) -> Graph:
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
    inputs = {"x": Tensor((batch, hidden))}
    return Graph(ops=ops, inputs=inputs, outputs=["linear2"])


if __name__ == "__main__":
    # Experiment 1: larger transformer MLP block
    g = make_linear_gelu_linear(batch=32, hidden=4096, ff=16384)

    hw_small = HardwareConfig(sram_bytes=1 * 1024 * 1024)  # 1MB
    hw_large = HardwareConfig(sram_bytes=8 * 1024 * 1024)  # 8MB

    for hw in (hw_small, hw_large):
        result = run_schedule_pass(g, hw)
        chosen = result.chosen_schedule.name
        chosen_cost = result.costs[chosen]
        print(f"\n=== Hardware SRAM={hw.sram_bytes} bytes ===")
        print(f"chosen schedule: {chosen}")
        print(f"dram bytes: {chosen_cost.dram.total_bytes} ({chosen_cost.dram.breakdown})")
        print(f"peak sram bytes: {chosen_cost.sram.peak_bytes} ({chosen_cost.sram.breakdown})")
        print(f"reason: {result.reason}")

    # Experiment 2 (optional): smaller block that fits everywhere
    g_small = make_linear_gelu_linear(batch=32, hidden=1024, ff=4096)
    for hw in (hw_small, hw_large):
        result = run_schedule_pass(g_small, hw)
        chosen = result.chosen_schedule.name
        chosen_cost = result.costs[chosen]
        print(f"\n--- Small graph, SRAM={hw.sram_bytes} bytes ---")
        print(f"chosen schedule: {chosen}")
        print(f"dram bytes: {chosen_cost.dram.total_bytes}")
        print(f"peak sram bytes: {chosen_cost.sram.peak_bytes}")
