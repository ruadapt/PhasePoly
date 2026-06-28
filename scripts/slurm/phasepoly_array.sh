#!/bin/bash -l
# PhasePoly array task — runs ONE circuit per --array index. Submit via:
#   bash scripts/run_phasepoly_pipeline.sh --folders ./benchmarks/...
# (which uses --export=NIL + an env file passed as $1; see scripts/run_phasepoly_pipeline.sh).
#
# Required env:
#   TASKLIST          one absolute .qasm path per line (line $SLURM_ARRAY_TASK_ID picked)
#   TAG               experiment tag, becomes results/<TAG>/<circuit>/
# Optional env:
#   PROFILES_JSON     per-circuit profile JSON (preferred for production)
#   ROUNDS_JSON       inline JSON list of round dicts; ignored if PROFILES_JSON is set
#   RESULTS_DIR       default: <repo>/results
#   TIMEOUT           default: 7200  (soft, recorded only)
#   HARD_TIMEOUT      default: 40000 (wrapper-enforced wall kill, ~11h)
#   REPO              default: this script's repo root
#   VENV              optional: path to a virtualenv to activate (.venv/bin/activate).
#                     If unset, falls back to <REPO>/.venv if present.
#
#SBATCH --job-name=phasepoly_array
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
if [ "${1:-}" != "" ]; then
  source "$1"
fi
: "${TASKLIST:?need TASKLIST=path/to/tasklist.txt (one qasm per line)}"
: "${TAG:?need TAG=phasepoly_full}"
: "${REPO:?need REPO=path/to/repo (set by run_phasepoly_pipeline.sh)}"
: "${RESULTS_DIR:=$REPO/results}"
: "${TIMEOUT:=40000}"
: "${HARD_TIMEOUT:=40000}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate the project's virtualenv if available. Honours an explicit $VENV,
# otherwise looks for $REPO/.venv. If neither exists, falls back to the
# system python3 already on PATH.
if [ -n "${VENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
elif [ -f "$REPO/.venv/bin/activate" ]; then
  source "$REPO/.venv/bin/activate"
fi

QASM=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKLIST")
[ -n "$QASM" ] || { echo "error: no QASM at line $SLURM_ARRAY_TASK_ID of $TASKLIST" >&2; exit 2; }
CIRCUIT=$(basename "$QASM" .qasm)
TAG_DIR="$RESULTS_DIR/$TAG"
OUT_DIR="$TAG_DIR/$CIRCUIT"
LOG_PATH="$TAG_DIR/log.txt"
mkdir -p "$TAG_DIR" "$OUT_DIR"

echo "[$(date)] node=$(hostname) task=$SLURM_ARRAY_TASK_ID circuit=$CIRCUIT cpus=$SLURM_CPUS_PER_TASK"

ROUNDS_FLAGS=()
if [ -n "${PROFILES_JSON:-}" ]; then
  ROUNDS_FLAGS+=(--profiles-json "$PROFILES_JSON")
elif [ -n "${ROUNDS_FILE:-}" ]; then
  # ROUNDS_FILE is preferred over ROUNDS_JSON because JSON commas confuse
  # Slurm's --export= parser. run_phasepoly_pipeline.sh writes the JSON to disk.
  ROUNDS_FLAGS+=(--rounds-json "$(cat "$ROUNDS_FILE")")
elif [ -n "${ROUNDS_JSON:-}" ]; then
  ROUNDS_FLAGS+=(--rounds-json "$ROUNDS_JSON")
fi

cd "$REPO"
python3 scripts/run_phasepoly_one.py \
  --circuit-name "$CIRCUIT" \
  --input-qasm "$QASM" \
  --output-dir "$OUT_DIR" \
  --tag "$TAG" \
  --log-path "$LOG_PATH" \
  --timeout "$TIMEOUT" \
  --hard-timeout "$HARD_TIMEOUT" \
  "${ROUNDS_FLAGS[@]}"
