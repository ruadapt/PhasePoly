# PhasePoly methods — summary

The single Python entry point is [`phasepoly_synthesize`](../src/phasepoly.py). This page documents what each axis means in the language of the paper, what `phasepoly_synthesize` actually does for each value, and which combinations are valid.

## Vocabulary (paper ↔ code)

| Paper term | What it means | Where it lives in code |
|---|---|---|
| **Phase Polynomial Co-Optimization** | Joint optimization of the phase-parity and output-parity networks via space-bounded A\* search with cost $f(n) = g(n) + h_1(n) + h_2(n)$. | `method="row_heap"` (or `row_heap_classical_GE`) |
| **Single-block Greedy Optimization** | Greedy CNOT synthesis applied independently per phase-polynomial block; baseline. | `method="single_block_greedy"` (or `single_block_greedy_classical_GE`) |
| **SSA-style Rotation Merging** | Whole-circuit rotation merging using SSA-style qubit-state renaming so $R_z$ gates merge across $H$ barriers when they target the same SSA ID. | `method="rotation_merging"` (search-free baseline), or applied as a pre/post pass on every other method via `rotation_merging_mode` |
| **Cross-block Intermediate Representation** | Joint phase-parity / output-parity matrix that spans several adjacent phase-polynomial blocks under a rank-based feasibility check. | enabled by `group_size ≥ 2` (requires a `row_heap*` method) |
| **Incremental Block Merging Strategy** | Multi-round schedule that starts with single-block optimization and progressively widens `group_size`. Successful rounds are chained; failed rounds keep using the last successful input. The in-file default is 5 rounds with `group_size = 1 → 3 → 1 → 5 → 1`; the paper's best-results config extends this to 7 rounds (`.../1/7/1`) with larger heap sizes — see `benchmarks/scripts/config/super_parameters_7200s.json`. | the default chained-rounds schedule in [`scripts/run_experiment.py`](../scripts/run_experiment.py) (`ROUNDS`) |
| **GF(2) Gaussian elimination backend** | Linear-system solver used both inside the A\* heuristic ($h_2$) and for the final output-parity reduction. Two variants. | `gaussian_elim_algorithm="modified"` (one-column lookahead pivot) or `"classic"` (greedy column-cost) |

## `phasepoly_synthesize` parameters

| Parameter | Type / values | Default | Effect |
|---|---|---|---|
| `circuit_input_path` | `str` (path) | required | Input QASM. |
| `circuit_output_path` | `str` (path) | required | Output QASM (parent dir is created if missing). |
| `method` | one of `row_heap`, `row_heap_classical_GE`, `single_block_greedy`, `single_block_greedy_classical_GE`, `pure_rotation_merging`, `rotation_merging` | **`row_heap`** | Synthesis strategy — see "method matrix" below. |
| `rotation_merging_mode` | `pure_rotation_merging` ∣ `advanced_rotation_merging` | **`advanced_rotation_merging`** | The SSA-style rotation-merging pass run before *and* after synthesis. The advanced variant additionally rewrites $\text{CNOT}\cdot H$ pairs and propagates $X$ gates through CNOT during preprocessing. |
| `heap_size` | `int ≥ 1` | **`10000`** | A\* priority-queue cap (max partial states alive). |
| `ends_checked` | `int ≥ 1` | **`10000`** | Multiple-solution budget $k$ — how many goal states to collect before returning the best one. |
| `group_size` | `int ≥ 1` | **`1`** | `1` = single-block; `≥ 2` = merge that many adjacent blocks via the cross-block IR before synthesis. Requires a `row_heap*` method. |
| `gaussian_elim_algorithm` | `modified` ∣ `classic` | **`modified`** | GF(2) Gaussian elimination backend. |
| `circuit_name`, `label` | `str` | `"circuit"`, `"0"` | Echoed back in the result; used as filename hints. |

## Validation rules (in `phasepoly_synthesize`)

- `method` outside the valid set → `ValueError`.
- `rotation_merging_mode` outside `{pure_rotation_merging, advanced_rotation_merging}` → `ValueError`.
- `gaussian_elim_algorithm` outside `{modified, classic}` → `ValueError`.
- `group_size > 1` with a non-`row_heap*` method → `ValueError`.

## Result fields (`PhasepolySynthesisResult`)

```python
success: bool
failure_stage: str          # 'read_input'|'rotation_merging_pre'|'partition'|'synthesis'|...
error_type, error: str
input_circuit_info, circuit_info: dict   # keys: gates(always weighted), weighted_cx, rz_gate, t_gate, ...
total_reduction, cx_reduction, rz_reduction, t_reduction: float
rotation_merging_time, circuit_partitioning_time, synthesis_time, data_io_time, total_time: float
warnings: list[str]
```

If synthesis raises, `phasepoly_synthesize` re-raises a `PhasepolySynthesisError` whose `.result` is the partially-populated `PhasepolySynthesisResult`.

## Practical recipes

| Goal | Suggested call |
|---|---|
| Fastest reasonable run | `method="single_block_greedy"` |
| Best quality, single block | `method="row_heap"`, `heap_size=10000`, `ends_checked=10000`, `group_size=1` |
| Cross-block reductions on $H$-heavy circuits | `method="row_heap"`, `group_size=3..7` |
| Compare GE backends | swap `gaussian_elim_algorithm` between `modified` and `classic`, hold the rest fixed |
| Rotation-merge-only baseline | `method="rotation_merging"` (advanced) or `"pure_rotation_merging"` (pure) |
| Production schedule (Incremental Block Merging) | call [`scripts/run_experiment.py`](../scripts/run_experiment.py) — default `ROUNDS` chains `group_size ∈ {1, 3, 1, 5, 1}`; the 7-round paper config is `benchmarks/scripts/config/super_parameters_7200s.json` |
