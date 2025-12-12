# MLCOMPLIER

Toy compiler pass that selects execution schedules for small transformer subgraphs (e.g., `Linear → GELU → Linear`) using a simple hardware model.

The pass estimates:

- **Memory traffic** (reads/writes to DRAM)
- **On‑chip SRAM usage** for intermediates

If all intermediates fit in SRAM, it prefers a memory‑aware/fused schedule; otherwise it falls back to a naive schedule. The same graph will compile differently under different hardware configurations, illustrating hardware–software co‑design.

## Repo layout

- `src/mlcompiler/` – minimal IR, hardware model, scheduling pass
- `examples/` – runnable toy graphs
- `tests/` – smoke tests

## Quickstart

Create a virtualenv and install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the example:

```bash
python examples/simple_transformer_subgraph.py
```

## Next steps

This is boilerplate scaffolding. The real pass logic, richer IR, and better cost models will be added.
