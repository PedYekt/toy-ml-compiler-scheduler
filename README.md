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

## Results

Running the same `Linear → GELU → Linear` subgraph under two SRAM sizes produces different schedules:

| Graph shape (batch, hidden, ff) | SRAM = 1MB | DRAM bytes | Peak SRAM | SRAM = 8MB | DRAM bytes | Peak SRAM |
|---|---:|---:|---:|---:|---:|---:|
| (32, 1024, 4096) | memory_aware | 128KB | 512KB | memory_aware | 128KB | 512KB |
| (32, 2048, 8192) | memory_aware | 256KB | 1MB | memory_aware | 256KB | 1MB |
| (32, 4096, 16384) | naive | 4.5MB | 1MB | memory_aware | 512KB | 2MB |

**What this demonstrates**

- The compiler makes **hardware‑aware schedule decisions** based on SRAM capacity.
- A larger graph flips from `naive` (materialize to DRAM) to `memory_aware` (keep intermediates on‑chip) when SRAM increases.
- Memory‑aware scheduling can **dramatically reduce DRAM traffic** when intermediates fit.
- This toy pass illustrates the core idea of **hardware–software co‑design**: the same IR compiles differently under different hardware configs.
