#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SLURM_SCRIPT:-}"
CHECK_ONLY=0
SMOKE_MODE=0

require_env_var() {
  local name="$1"
  local value="${!name:-}"
  if [ -z "$value" ]; then
    echo "ERROR: $name must be set before launch."
    exit 1
  fi
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -smoke|--smoke)
      SMOKE_MODE=1
      shift
      ;;
    --check)
      CHECK_ONLY=1
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

if [ "${#POSITIONAL[@]}" -lt 1 ]; then
  echo "ERROR: experiment directory required (contains runs.json and slurm.json)"
  exit 1
fi
EXP_DIR="${POSITIONAL[0]}"
if [ ! -d "$EXP_DIR" ]; then
  echo "ERROR: experiment directory not found: $EXP_DIR"
  exit 1
fi

EXP_DIR="$(cd "$EXP_DIR" && pwd)"
RUNS_FILE="${EXP_DIR}/runs.json"
SLURM_CONFIG_PATH="${EXP_DIR}/slurm.json"

if [ ! -f "$RUNS_FILE" ]; then
  echo "ERROR: Runs file not found: $RUNS_FILE"
  exit 1
fi
if [[ "$RUNS_FILE" != *.json ]]; then
  echo "ERROR: Runs file must be JSON: $RUNS_FILE"
  exit 1
fi

SLURM_SCRIPT="${SLURM_SCRIPT:-${SCRIPT_DIR}/ray_on_slurm.slurm}"

if [ ! -f "$SLURM_SCRIPT" ]; then
  echo "ERROR: SLURM_SCRIPT not found: $SLURM_SCRIPT"
  exit 1
fi

if [ ! -f "$SLURM_CONFIG_PATH" ]; then
  echo "ERROR: slurm.json not found: $SLURM_CONFIG_PATH"
  exit 1
fi

python3 "${SCRIPT_DIR}/slurm_config_utils.py" validate-config --config "$SLURM_CONFIG_PATH"

require_env_var "SCRATCH_DIR"
require_env_var "ARCHIVE_RESULTS_ROOT"
require_env_var "ARCHIVE_LOGS_ROOT"

NUM_LINES=$(python3 "${SCRIPT_DIR}/config_utils.py" count --config "$RUNS_FILE")

if [ "$NUM_LINES" -lt 1 ]; then
  echo "ERROR: No runs found in $RUNS_FILE"
  exit 1
fi

SBATCH_ARGS=()
MAIL_ARGS=()
if [ -n "${SLURM_CONFIG_PATH:-}" ] && [ -f "${SLURM_CONFIG_PATH:-}" ]; then
readarray -t SBATCH_ARGS < <(python3 "${SCRIPT_DIR}/slurm_config_utils.py" sbatch-args --config "$SLURM_CONFIG_PATH")
  readarray -t MAIL_ARGS < <(python3 "${SCRIPT_DIR}/slurm_config_utils.py" mail-args --config "$SLURM_CONFIG_PATH")
fi

# Auto-enable exclusive node allocation for 2-GPU jobs unless already requested.
# Skip this in smoke mode to improve scheduling latency for quick checks.
REQUESTED_GPUS_PER_NODE=""
HAS_EXCLUSIVE_FLAG=0
for ((i=0; i<${#SBATCH_ARGS[@]}; i++)); do
  arg="${SBATCH_ARGS[$i]}"
  case "$arg" in
    --exclusive)
      HAS_EXCLUSIVE_FLAG=1
      ;;
    --gpus-per-node=*)
      REQUESTED_GPUS_PER_NODE="${arg#*=}"
      ;;
    --gpus=*)
      REQUESTED_GPUS_PER_NODE="${arg#*=}"
      ;;
    --gres=*)
      gres="${arg#*=}"
      if [[ "$gres" == gpu:* ]]; then
        REQUESTED_GPUS_PER_NODE="${gres#gpu:}"
      fi
      ;;
    --gres)
      next="${SBATCH_ARGS[$((i+1))]:-}"
      if [[ "$next" == gpu:* ]]; then
        REQUESTED_GPUS_PER_NODE="${next#gpu:}"
      fi
      ;;
  esac
done

if [ "$SMOKE_MODE" -eq 0 ] && [ "$HAS_EXCLUSIVE_FLAG" -eq 0 ] && [ "$REQUESTED_GPUS_PER_NODE" = "2" ]; then
  SBATCH_ARGS+=(--exclusive)
  echo "INFO: Auto-applying --exclusive for 2-GPU request."
elif [ "$SMOKE_MODE" -eq 1 ] && [ "$REQUESTED_GPUS_PER_NODE" = "2" ]; then
  echo "INFO: Smoke mode enabled; skipping auto --exclusive for 2-GPU request."
fi

EXPORT_VARS="ALL,RUNS_FILE=$RUNS_FILE,SUBMIT_SCRIPT_DIR=$SCRIPT_DIR,SCRATCH_DIR=${SCRATCH_DIR},ARCHIVE_RESULTS_ROOT=${ARCHIVE_RESULTS_ROOT},ARCHIVE_LOGS_ROOT=${ARCHIVE_LOGS_ROOT}"
if [ -n "${SLURM_CONFIG_PATH:-}" ]; then
  EXPORT_VARS="${EXPORT_VARS},SLURM_CONFIG_PATH=${SLURM_CONFIG_PATH}"
fi
if [ -n "${VENV_ACTIVATE:-}" ]; then
  EXPORT_VARS="${EXPORT_VARS},VENV_ACTIVATE=${VENV_ACTIVATE}"
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  echo "CHECK: running local preflight for $NUM_LINES run(s) with $SLURM_SCRIPT"
  REQUESTED_GPUS_PER_NODE=""
  REQUESTED_CPUS_PER_TASK=""
  for ((i=0; i<${#SBATCH_ARGS[@]}; i++)); do
    arg="${SBATCH_ARGS[$i]}"
    case "$arg" in
      --gpus-per-node=*)
        REQUESTED_GPUS_PER_NODE="${arg#*=}"
        ;;
      --gpus=*)
        REQUESTED_GPUS_PER_NODE="${arg#*=}"
        ;;
      --gres=*)
        gres="${arg#*=}"
        if [[ "$gres" == gpu:* ]]; then
          REQUESTED_GPUS_PER_NODE="${gres#gpu:}"
        fi
        ;;
      --gres)
        next="${SBATCH_ARGS[$((i+1))]:-}"
        if [[ "$next" == gpu:* ]]; then
          REQUESTED_GPUS_PER_NODE="${next#gpu:}"
        fi
        ;;
      --cpus-per-task=*)
        REQUESTED_CPUS_PER_TASK="${arg#*=}"
        ;;
      --cpus-per-task)
        REQUESTED_CPUS_PER_TASK="${SBATCH_ARGS[$((i+1))]:-}"
        ;;
      -c)
        REQUESTED_CPUS_PER_TASK="${SBATCH_ARGS[$((i+1))]:-}"
        ;;
      -c[0-9]*)
        REQUESTED_CPUS_PER_TASK="${arg#-c}"
        ;;
    esac
  done
  if [ -z "$REQUESTED_GPUS_PER_NODE" ]; then
    REQUESTED_GPUS_PER_NODE=1
  fi
  if [ -z "$REQUESTED_CPUS_PER_TASK" ]; then
    REQUESTED_CPUS_PER_TASK="$(nproc)"
    echo "CHECK: cpus-per-task not found in sbatch args; defaulting to nproc=$REQUESTED_CPUS_PER_TASK"
  fi
  echo "CHECK: using REQUESTED_GPUS_PER_NODE=$REQUESTED_GPUS_PER_NODE"
  echo "CHECK: using REQUESTED_CPUS_PER_TASK=$REQUESTED_CPUS_PER_TASK"

  for idx in $(seq 1 "$NUM_LINES"); do
    echo "CHECK: preflight run $idx/$NUM_LINES"
    SUBMIT_SCRIPT_DIR="$SCRIPT_DIR" \
      RUNS_FILE="$RUNS_FILE" \
      SLURM_ARRAY_TASK_ID="$idx" \
      SLURM_GPUS_PER_NODE="$REQUESTED_GPUS_PER_NODE" \
      SLURM_CPUS_PER_TASK="$REQUESTED_CPUS_PER_TASK" \
      PREFLIGHT_ONLY=1 \
      bash "$SLURM_SCRIPT"
  done
  echo "CHECK: OK"
  exit 0
fi

sbatch_output=$(sbatch "${SBATCH_ARGS[@]}" "${MAIL_ARGS[@]}" \
  --parsable \
  --array=1-"$NUM_LINES" \
  --export="$EXPORT_VARS" \
  "$SLURM_SCRIPT")
job_id="${sbatch_output%%;*}"
echo "Submitted batch job ${job_id}"
