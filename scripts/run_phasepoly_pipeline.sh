#!/usr/bin/env bash
# run_phasepoly_pipeline.sh — submit one Slurm sbatch array that runs PhasePoly
# on every .qasm file in the supplied benchmark folders. Same script for
# grayscale tests (~10-15 s budgets) and full production runs (hours per
# circuit).
#
# This is the only sbatch entrypoint. It builds the tasklist, derives sensible
# timeouts, and submits with --array=1-N%K plus an --export=NIL envelope so
# each array task lands in scripts/slurm/phasepoly_array.sh and runs ONE
# circuit through scripts/run_phasepoly_one.py.
#
# Output:
#   results/<TAG>/                     # per-tag output dir (phasepoly_best/, log.txt, ...)
#   results/<TAG>/sbatch_logs/         # Slurm stdout/err per array task
#                                      # filename: phasepoly_<tag>-<jobid>_<taskid>.{out,err}
#
# === USAGE ==================================================================
#
#   bash scripts/run_phasepoly_pipeline.sh \
#     --folders <comma-separated-paths> \   # accepts EITHER:
#                                           #   • names under <repo>/benchmarks/  (flat or nested, e.g. general,larger_circuits/adder)
#                                           #   • paths starting with /, ~, ., or .. → taken literally (relative to cwd or absolute)
#                                           #   e.g. --folders ./my_circuits,/abs/path/qasm,larger_circuits/mcx
#     [--circuits a,b,c] \
#     [--profiles-json PATH] \              # per-circuit profile lookup
#     [--rounds-json JSON] \                # inline rounds list (mutex w/ profiles)
#     [--timeout SEC] \                     # soft timeout, recorded only (default = HARD_TIMEOUT)
#     [--hard-timeout SEC] \                # wall-clock kill (default 40000 ≈ 11h)
#     [--jobs N] \                          # %N concurrency cap (default 16)
#     [--tag NAME] \                        # default: phasepoly_<timestamp>
#     [--mem GB] \                          # default 40
#     [--time HH:MM:SS] \                   # default = HARD_TIMEOUT + 30 min slack
#     [--repo PATH] \                       # default: this script's repo root
#     [--venv PATH] \                       # virtualenv to activate inside the array task; defaults to <repo>/.venv if present
#     [--tasklist-out PATH] \               # default: ~/jobs/phasepoly_<tag>_tasklist.txt
#     [--dry-run] \                         # print sbatch command, don't submit
#     [--wait]                              # block until the array finishes
#
# === GRAYSCALE EXAMPLE (10 s soft / 20 s hard, ~30 s wall per task) =========
#
#   bash scripts/run_phasepoly_pipeline.sh \
#       --folders general,larger_circuits/adder,larger_circuits/hwb,larger_circuits/mcx \
#       --hard-timeout 20 --tag phasepoly_grayscale --wait
#
# === PRODUCTION EXAMPLE =====================================================
#
#   bash scripts/run_phasepoly_pipeline.sh \
#       --folders general,larger_circuits/adder,larger_circuits/hwb,larger_circuits/mcx \
#       --hard-timeout 40000 --tag phasepoly_full

set -uo pipefail

usage() { sed -n '2,/^$/p' "$0" | sed 's/^# \?//' | sed -n '/USAGE/,/PRODUCTION EXAMPLE/p'; exit 0; }
die()   { echo "error: $*" >&2; exit 2; }

# --- defaults ------------------------------------------------------------
FOLDERS=""
CIRCUITS=""
PROFILES_JSON=""
ROUNDS_JSON=""
TIMEOUT=""
HARD_TIMEOUT=""
JOBS=16
TAG=""
MEM=40
WALLTIME=""           # set after HARD_TIMEOUT is known
REPO_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$REPO_DEFAULT"
VENV=""
TASKLIST_OUT=""
DRY_RUN=0
WAIT=0

# --- argparse ------------------------------------------------------------
while [ "$#" -gt 0 ]; do
  case "$1" in
    --folders)          FOLDERS="${2:-}"; shift 2 ;;
    --circuits)         CIRCUITS="${2:-}"; shift 2 ;;
    --profiles-json)    PROFILES_JSON="${2:-}"; shift 2 ;;
    --rounds-json)      ROUNDS_JSON="${2:-}"; shift 2 ;;
    --timeout)          TIMEOUT="${2:-}"; shift 2 ;;
    --hard-timeout)     HARD_TIMEOUT="${2:-}"; shift 2 ;;
    --jobs)             JOBS="${2:-}"; shift 2 ;;
    --tag)              TAG="${2:-}"; shift 2 ;;
    --mem)              MEM="${2:-}"; shift 2 ;;
    --time)             WALLTIME="${2:-}"; shift 2 ;;
    --repo)             REPO="${2:-}"; shift 2 ;;
    --venv)             VENV="${2:-}"; shift 2 ;;
    --tasklist-out)     TASKLIST_OUT="${2:-}"; shift 2 ;;
    --dry-run)          DRY_RUN=1; shift ;;
    --wait)             WAIT=1; shift ;;
    -h|--help)          usage ;;
    *) die "unknown arg: $1" ;;
  esac
done

# --- validation + defaults -----------------------------------------------
[ -n "$FOLDERS" ] || die "--folders is required.
  Names resolve under <repo>/benchmarks/        (e.g. general,larger_circuits/adder)
  Paths starting with / ~ . or .. are literal   (e.g. ./my_qasm,/data/circuits,../shared/foo)
  Mix is fine                                   (e.g. general,./extra)"
[ -n "$REPO" ] && [ -d "$REPO" ] || die "--repo not a directory: $REPO"

: "${HARD_TIMEOUT:=40000}"
: "${TIMEOUT:=$HARD_TIMEOUT}"      # soft timeout is recorded only

[ -z "${TAG}" ] && TAG="phasepoly_$(date +%Y%m%d_%H%M%S)"
[ -z "${TASKLIST_OUT}" ] && TASKLIST_OUT="$HOME/jobs/phasepoly_${TAG}_tasklist.txt"

# WALLTIME = HARD_TIMEOUT + 30min slack. Slurm's --time is the wall-clock kill;
# if it's smaller than our wrapper's HARD_TIMEOUT, Slurm kills the task before
# the wrapper can hard-timeout cleanly and write a summary row.
if [ -z "$WALLTIME" ]; then
  total_secs=$(( HARD_TIMEOUT + 1800 ))
  WALLTIME=$(printf '%02d:%02d:00' $(( total_secs / 3600 )) $(( (total_secs % 3600) / 60 )))
fi
mkdir -p "$(dirname "$TASKLIST_OUT")"

# Mutually exclusive rounds source
if [ -n "$PROFILES_JSON" ] && [ -n "$ROUNDS_JSON" ]; then
  die "--profiles-json and --rounds-json are mutually exclusive"
fi

# --- build tasklist -------------------------------------------------------
BUILD_ARGS=(--benchmark-folders "$FOLDERS" --output "$TASKLIST_OUT")
[ -n "$CIRCUITS" ]      && BUILD_ARGS+=(--circuits "$CIRCUITS")
# With a profiles file, restrict tasklist to circuits actually listed there.
[ -n "$PROFILES_JSON" ] && BUILD_ARGS+=(--profiles-json "$PROFILES_JSON")

echo "[run_phasepoly_pipeline] building tasklist -> $TASKLIST_OUT"
N=$(python3 "$REPO/scripts/slurm/build_tasklist.py" "${BUILD_ARGS[@]}")
[ "$N" -gt 0 ] || die "tasklist is empty"
echo "[run_phasepoly_pipeline] tasks: $N"

# --- assemble tiny Slurm env file -----------------------------------------
# Avoid --export=ALL: a large user shell env (long PATH / many vars) can trip
# Slurm's "user env retrieval failed requeued held" with batch tasks silently
# parked in PD. Write only the variables our array.sh reads and submit with
# --export=NIL, then let the array script source this file.
SBATCH_PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
EXPORTS=("HOME=$HOME" "USER=$USER" "PATH=$SBATCH_PATH" "TASKLIST=$TASKLIST_OUT" "TAG=$TAG" "TIMEOUT=$TIMEOUT" "HARD_TIMEOUT=$HARD_TIMEOUT" "REPO=$REPO")
[ -n "$VENV" ] && EXPORTS+=("VENV=$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$VENV")")
[ -n "$PROFILES_JSON" ] && EXPORTS+=("PROFILES_JSON=$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$PROFILES_JSON")")
if [ -n "$ROUNDS_JSON" ]; then
  # Slurm --export=A,B,C parses commas itself, so a comma-bearing JSON value
  # gets sliced. Stash it in a file and pass the file path instead.
  ROUNDS_FILE="$HOME/jobs/phasepoly_${TAG}_rounds.json"
  printf '%s' "$ROUNDS_JSON" > "$ROUNDS_FILE"
  EXPORTS+=("ROUNDS_FILE=$ROUNDS_FILE")
fi
SBATCH_FILE="$REPO/scripts/slurm/phasepoly_array.sh"

SBATCH_LOG_DIR="$REPO/results/$TAG/sbatch_logs"
mkdir -p "$SBATCH_LOG_DIR"
ENV_FILE="$REPO/results/$TAG/job_env.sh"
python3 "$REPO/scripts/slurm/write_sbatch_env.py" \
  --output "$ENV_FILE" \
  "${EXPORTS[@]}"

# Job name carries TAG so log filenames are self-identifying:
#   results/<TAG>/sbatch_logs/phasepoly_<tag>-<arrayJobId>_<taskId>.out
JOB_NAME="phasepoly_${TAG}"

SBATCH_CMD=(sbatch
  "--job-name=$JOB_NAME"
  "--array=1-${N}%${JOBS}"
  "--time=$WALLTIME"
  "--mem=${MEM}G"
  "--output=$SBATCH_LOG_DIR/%x-%A_%a.out"
  "--error=$SBATCH_LOG_DIR/%x-%A_%a.err"
  "--export=NIL"
  "$SBATCH_FILE"
  "$ENV_FILE")

echo "[run_phasepoly_pipeline] submission:"
printf '  %q ' "${SBATCH_CMD[@]}"; echo

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[run_phasepoly_pipeline] --dry-run set; not submitting."
  exit 0
fi

# --- submit ---------------------------------------------------------------
SBATCH_OUT=$("${SBATCH_CMD[@]}") || die "sbatch failed"
echo "[run_phasepoly_pipeline] $SBATCH_OUT"
JOB_ID=$(echo "$SBATCH_OUT" | awk '{print $4}')
[ -n "$JOB_ID" ] || die "could not parse job id from sbatch output"

echo "[run_phasepoly_pipeline] job=$JOB_ID  tag=$TAG"
echo "[run_phasepoly_pipeline] tasklist: $TASKLIST_OUT"
echo "[run_phasepoly_pipeline] results:  $REPO/results/$TAG/"

# --- persist job_info.txt + append to master log -------------------------
JOB_INFO_FILE="$REPO/results/$TAG/job_info.txt"
cat > "$JOB_INFO_FILE" <<EOF
# Slurm submission record for tag '$TAG'
# Generated by run_phasepoly_pipeline.sh at $(date '+%Y-%m-%d %H:%M:%S %Z')

JOB_ID=$JOB_ID
TAG=$TAG
ARRAY_SIZE=$N
CONCURRENCY=%$JOBS
TIMEOUT=$TIMEOUT
HARD_TIMEOUT=$HARD_TIMEOUT
TASKLIST=$TASKLIST_OUT
RESULTS_DIR=$REPO/results/$TAG
ENV_FILE=$ENV_FILE

# === Check whether the job is still running ===
squeue -h -u \$USER -j $JOB_ID | wc -l        # 0 = done; nonzero = tasks still in queue
sacct -j $JOB_ID -P -o JobID,State,Elapsed,MaxRSS -n | awk -F'|' '\$1 ~ /_[0-9]+\$/{c[\$2]++} END{for(k in c) print k": "c[k]}'

# === Cancel the entire job (or a single array task) ===
scancel $JOB_ID                  # the whole array
scancel ${JOB_ID}_<task-index>   # one specific task, e.g. ${JOB_ID}_5

# === When done, look at results ===
ls $REPO/results/$TAG/
cat $REPO/results/$TAG/phasepoly_best/summary.csv
EOF
echo "[run_phasepoly_pipeline] saved   : $JOB_INFO_FILE"

# Master submissions log (one line per submission, for cross-tag bookkeeping)
MASTER_LOG="$REPO/results/_submissions.log"
mkdir -p "$REPO/results"
printf '%s | job=%s | tag=%s | array=1-%s%%%s | results=%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$JOB_ID" "$TAG" "$N" "$JOBS" \
  "$REPO/results/$TAG" >> "$MASTER_LOG"

if [ "$WAIT" -eq 0 ]; then
  echo "[run_phasepoly_pipeline] not blocking; squeue -j $JOB_ID to monitor."
  exit 0
fi

# --- wait + summarize -----------------------------------------------------
echo "[run_phasepoly_pipeline] waiting for job $JOB_ID..."
while squeue -h -j "$JOB_ID" 2>/dev/null | grep -q .; do
  sleep 5
done
echo "[run_phasepoly_pipeline] job $JOB_ID done; summary:"
sacct -j "$JOB_ID" -P -o JobID,State,Elapsed,MaxRSS,ExitCode 2>/dev/null \
  | column -ts'|' || echo "(sacct unavailable)"

RESULTS="$REPO/results/$TAG"
echo
echo "[run_phasepoly_pipeline] artifacts in $RESULTS:"
wait_for_nonempty() {
  local file=$1
  local attempt
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    [ -s "$file" ] && return 0
    sleep 1
  done
  [ -s "$file" ]
}
SUMMARY="$RESULTS/phasepoly_best/summary.csv"
if wait_for_nonempty "$SUMMARY"; then
  ROWS=$(($(wc -l < "$SUMMARY")-1))
  echo "  phasepoly_best/summary.csv: $ROWS row(s)"
  head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | head -5
else
  echo "  WARN: $SUMMARY missing/empty"
fi
