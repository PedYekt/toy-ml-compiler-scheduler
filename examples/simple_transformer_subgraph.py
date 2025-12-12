from mlcompiler import Graph, HardwareConfig, Node, OpKind, TensorType, choose_schedule


def make_linear_gelu_linear(batch: int = 8, hidden: int = 1024, intermediate: int = 4096) -> Graph:
    x = "x"
    n1 = Node(
        name="linear1",
        op=OpKind.LINEAR,
        inputs=[x],
        output=TensorType((batch, intermediate)),
    )
    n2 = Node(
        name="gelu",
        op=OpKind.GELU,
        inputs=[n1.name],
        output=TensorType((batch, intermediate)),
    )
    n3 = Node(
        name="linear2",
        op=OpKind.LINEAR,
        inputs=[n2.name],
        output=TensorType((batch, hidden)),
    )
    return Graph(nodes=[n1, n2, n3], outputs=[n3.name], inputs=[x])


if __name__ == "__main__":
    g = make_linear_gelu_linear()
    tiny = HardwareConfig(name="tiny_sram", sram_bytes=256 * 1024)
    big = HardwareConfig(name="big_sram", sram_bytes=8 * 1024 * 1024)

    for hw in (tiny, big):
        choice = choose_schedule(g, hw)
        print(f"{hw.name}: {choice.schedule.kind} (intermediates={choice.estimated_intermediate_bytes} bytes)")

