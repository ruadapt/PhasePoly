# Running experiments

Three execution modes, increasing in scale:

1. [Single-circuit driver](#1-single-circuit-driver) — one QASM, one process, structured output, wall-clock timeout.
2. [Serial batch](#2-serial-batch-driver) — folder of circuits, multi-round chained schedule (Incremental Block Merging), runs locally.
3. [Slurm fan-out](#3-slurm-fan-out) — one cluster array; covered in [slurm.md](slurm.md).

All three end up writing to `results/<TAG>/` with the same layout (see [Output layout](#output-layout)).

---

## 1. Single-circuit driver

`scripts/run_phasepoly_one.py` is the unit of work for everything else. It spawns `_run_one_circuit.py` under a wall-clock timeout, recovers `(best).qasm` if killed, and upserts a row in `phasepoly_best_<tag>/summary.csv`.

### Inline rounds

```bash
python scripts/run_phasepoly_one.py \
    --circuit-name adder_8 \
    --input-qasm benchmarks/general/adder_8.qasm \
    --output-dir results/quick/adder_8 \
    --tag quick \
    --log-path results/quick/log.txt \
    --rounds-json '[
        {"method":"row_heap","heap_size":1000,"ends_checked":1000,"group_size":1},
        {"method":"row_heap","heap_size":1000,"ends_checked":1000,"group_size":3}
    ]' \
    --timeout 600 --hard-timeout 600
```

Round N+1 reads round N's output. Each round is a dict with any subset of the keys allowed by `phasepoly_synthesize` (see [methods_summary.md](methods_summary.md)).

### Profile-based rounds

```bash
python scripts/run_phasepoly_one.py \
    --circuit-name adder_8 \
    --input-qasm benchmarks/general/adder_8.qasm \
    --output-dir results/big/adder_8 \
    --tag big \
    --log-path results/big/log.txt \
    --profiles-json /path/to/your_profiles.json \
    --timeout 7200 --hard-timeout 9000
```

A profiles JSON file defines named profiles (e.g. `g1`, `g3`, `g1_500`, `g7`) and per-circuit `assign` arrays — the per-circuit rounds list. Useful when the same circuit needs different schedules at different scales.

---

## 2. Serial batch driver

`scripts/run_experiment.py` enumerates a folder, runs the chained `ROUNDS` schedule on each circuit. The in-file default is the **Incremental Block Merging** strategy: 5 rounds with `group_size ∈ {1, 3, 1, 5, 1}` and `heap_size = ends_checked = 1000` throughout — start single-block, widen the merge window, refine again. Each successful round feeds the next round; failed rounds keep the last successful output. The paper's best-results schedule (7 rounds, extending to `.../1/7/1` with per-circuit heap sizes) is `benchmarks/scripts/config/super_parameters_7200s.json` — load it via `--profiles-json`.

```bash
# Quick local benchmark:
python scripts/run_experiment.py --tag quickstart \
    --circuits barenco_tof_4,tof_3 \
    --timeout 120 \
    --rounds-json '[{"method":"row_heap","heap_size":1000,"ends_checked":1000,"group_size":1}]'

# In-file defaults (benchmarks/general/, tag=exp_default):
python scripts/run_experiment.py

# Custom tag and circuit subset:
python scripts/run_experiment.py --tag adder_run --circuits adder_8,grover_5

# Inline rounds (overrides the in-file ROUNDS):
python scripts/run_experiment.py --tag greedy_only \
    --rounds-json '[{"method":"single_block_greedy"}]'
```

Common flags:

| Flag | Default | Note |
|---|---|---|
| `--tag NAME` | `exp_default` | `results/<TAG>/` identifier |
| `--input-dir DIR` | `./benchmarks/general/` | folder of `.qasm` files |
| `--circuits a,b,c` | all in `--input-dir` | filter by stem |
| `--timeout SEC` | `600` | per-circuit wall-clock budget across all rounds |
| `--rounds-json JSON` | (uses in-file `ROUNDS`) | inline schedule |
| `--profiles-json PATH` | (uses in-file `ROUNDS`) | per-circuit profile lookup |

The driver runs circuits one at a time in process. For parallelism, use the Slurm path.

For the quick local benchmark, inspect `results/quickstart/phasepoly_best_quickstart/summary.csv` and the `*(best).qasm` files in the same directory.

---

## 3. Slurm fan-out

Covered in [slurm.md](slurm.md). One sbatch array via `scripts/run_phasepoly_pipeline.sh`.

---

## Output layout

```text
results/<TAG>/
├── log.txt                              # multi-process append-safe log
├── job_info.txt                         # squeue/sacct/scancel cheat sheet (Slurm only)
├── sbatch_logs/phasepoly_<tag>-<jid>_<task>.{out,err}   # Slurm only
├── phasepoly_best_<TAG>/
│   ├── summary.csv                      # circuit_name, model, is_timeout, weighted_cx, ...
│   ├── <circuit>(best).qasm
│   └── <circuit>(best).txt
└── <circuit>/                           # per-circuit subdir
    ├── <circuit>(N).qasm                # round-N output (chained input for round N+1)
    ├── <circuit>(N).txt                 # round-N metrics
    ├── <circuit>(N).json                # round-N machine-readable
    └── _progress.json                   # checkpoint (recovers (best).qasm if killed)
```

`summary.csv` schema:

```text
circuit_name, model, is_timeout, total_gate_count, weighted_cx, rz_gate_count,
t_gate_count, weighted_depth, synthesis_time, total_time
```

`_phasepoly_best.update_phasepoly_best()` upserts rows with `fcntl.flock`, so concurrent Slurm tasks won't corrupt the file.

## Retries

```bash
MISSING=$(python scripts/find_missing.py --tag <TAG>)
# then re-run only those circuits with --circuits "$MISSING"
```

`find_missing.py` reads `phasepoly_best_*/summary.csv` and prints a comma-separated list of circuits that didn't finish.
