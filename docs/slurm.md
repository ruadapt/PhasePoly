# Slurm operations

## One sbatch entry point

`scripts/run_phasepoly_pipeline.sh` builds a tasklist and submits one `--array=1-N%K`. Same script for grayscale (10 s budgets) and full production (hours). One array task = one circuit.

```bash
bash scripts/run_phasepoly_pipeline.sh \
    --folders <comma-separated-paths> \
    [--circuits a,b,c] \
    [--profiles-json PATH | --rounds-json JSON] \
    [--timeout SEC] [--hard-timeout SEC] \
    [--jobs N] [--mem GB] [--time HH:MM:SS] \
    [--venv PATH] \
    [--tag NAME] [--wait | --dry-run]
```

`bash scripts/run_phasepoly_pipeline.sh -h` lists every flag.

## Two-segment timeout

| Knob | What it does |
|---|---|
| `--timeout SEC` | Soft, recorded only — the per-round cap lives inside the rounds JSON. |
| `--hard-timeout SEC` | Wall-clock kill enforced by the Python wrapper. On expiry the wrapper recovers the latest `(best).qasm` from `_progress.json`, marks `is_timeout=true` in the summary, and returns cleanly. |
| sbatch `--time HH:MM:SS` | Slurm-level kill, set well above `--hard-timeout` (default = `--hard-timeout` + 30 min). |

## Defaults the runner picks for you

| Knob | Default |
|---|---|
| Python environment | `<repo>/.venv` if it exists; otherwise system `python3`. Override with `--venv PATH`. |
| `--hard-timeout` | `40000` (≈ 11 h) |
| `--timeout` | same as `--hard-timeout` |
| `--mem` | `40G` |
| `--jobs` (concurrency cap) | `16` |

## sbatch template (what gets submitted)

```text
sbatch --job-name=phasepoly_<tag>
       --export=NIL                            # avoid serializing the user env
       --array=1-N%K                           # N tasks, %K concurrency cap
       --time=<HARD_TIMEOUT+30min> --mem=40G
       --output=results/<TAG>/sbatch_logs/<...>.out
       --error= results/<TAG>/sbatch_logs/<...>.err
       scripts/slurm/phasepoly_array.sh <env-file>
```

`phasepoly_array.sh` sources the env file, activates the virtualenv (`$VENV/bin/activate`, or `<repo>/.venv/bin/activate` as a fallback), reads the task at line `$SLURM_ARRAY_TASK_ID` from `TASKLIST`, and execs `python scripts/run_phasepoly_one.py --... --hard-timeout <N>`.

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Many tasks `PD (user env retrieval failed requeued held)` | Slurm couldn't serialize a giant user env (we already pass `--export=NIL`, but a `~/.bashrc`-injected env can still trip controllers) | `scontrol release <jobid>`. If persistent, source an explicit env file in `scripts/slurm/phasepoly_array.sh`. |
| `ModuleNotFoundError: networkx` | virtualenv not activated inside the array task | pass `--venv /abs/path/.venv` (or place a `.venv/` at the repo root and let the default kick in). |
| `status=hard-timeout` on big circuits | algorithm genuinely exceeds the budget | bump `--hard-timeout`; some MCX / QCLA / QAOA need ≥ 8 h. |
| `OOM-killed` | 40 GB default exceeded by a deep search on a 20+ qubit circuit | `--mem 80` or larger. |

## Monitoring a submission

```bash
cat results/<TAG>/job_info.txt        # ready-to-paste squeue / sacct / scancel for this submission
cat results/_submissions.log          # cross-tag history of every sbatch ever submitted
squeue -u $USER                       # live queue
```

`--wait` blocks until the array finishes and prints a `sacct` summary. `--dry-run` prints the sbatch command without submitting.

## Retries

After a partial run:

```bash
MISSING=$(python scripts/find_missing.py --tag <TAG>)
bash scripts/run_phasepoly_pipeline.sh --folders <same as before> \
    --circuits "$MISSING" --tag <TAG>_retry
```

`find_missing.py` reads `phasepoly_best_<TAG>/summary.csv` and prints a comma-separated list of circuits with no row or with `status ≠ ok`.
