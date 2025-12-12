from mlcompiler import Graph, HardwareConfig, Op, Tensor, choose_schedule


def make_linear_gelu_linear(batch: int = 8, hidden: int = 1024, intermediate: int = 4096) -> Graph:
    ops = [
        Op(
            name="Linear",
            inputs=["x"],
            outputs=["linear1"],
            attrs={"in_features": hidden, "out_features": intermediate},
        ),
        Op(name="GELU", inputs=["linear1"], outputs=["gelu"], attrs={}),
        Op(
            name="Linear",
            inputs=["gelu"],
            outputs=["linear2"],
            attrs={"in_features": intermediate, "out_features": hidden},
        ),
    ]
    inputs = {"x": Tensor((batch, hidden))}
    return Graph(ops=ops, inputs=inputs, outputs=["linear2"])


if __name__ == "__main__":
    g = make_linear_gelu_linear()
    tiny = HardwareConfig(sram_bytes=256 * 1024)
    big = HardwareConfig(sram_bytes=8 * 1024 * 1024)

    for hw in (tiny, big):
        choice = choose_schedule(g, hw)
        print(f"{hw.sram_bytes}B SRAM: {choice.schedule.kind} (intermediates={choice.estimated_intermediate_bytes} bytes)")
