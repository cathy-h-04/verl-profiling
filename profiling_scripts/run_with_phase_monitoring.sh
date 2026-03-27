#!/bin/bash
#
# run_with_phase_monitoring.sh
# Runs verl training with phase-level GPU monitoring

set -e

# Ensure SCRATCH_DIR is set for the cluster
: "${SCRATCH_DIR:?ERROR: SCRATCH_DIR must be set to a shared writable scratch directory before launch.}"
export SCRATCH_DIR
mkdir -p "$SCRATCH_DIR/logs" "$SCRATCH_DIR/checkpoints" "$SCRATCH_DIR/data"
if [ -n "${RAY_ADDRESS:-}" ]; then
    export RAY_ADDRESS
fi
# Prefer node-local scratch for temp/cache to avoid NFS stale file handles.
LOCAL_SCRATCH="${LOCAL_SCRATCH:-${SLURM_TMPDIR:-/scratch/${USER}}}"
if [ ! -d "$LOCAL_SCRATCH" ]; then
    LOCAL_SCRATCH="/tmp/${USER}"
fi
mkdir -p "$LOCAL_SCRATCH"
export TMPDIR="${TMPDIR:-${LOCAL_SCRATCH}/tmp}"
mkdir -p "$TMPDIR"
export RAY_TMPDIR="${RAY_TMPDIR:-${LOCAL_SCRATCH}/ray}"
mkdir -p "$RAY_TMPDIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LOCAL_SCRATCH}/xdg_cache}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-${LOCAL_SCRATCH}/vllm_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${LOCAL_SCRATCH}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_SCRATCH}/triton}"
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# -------------------- Arguments / Env --------------------
BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-${1:-gsm8k_phased}}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-${2:-1}}"
POLL_INTERVAL="${POLL_INTERVAL:-${3:-1}}"
GRANULARITY="${GRANULARITY:-${4:-phase}}"  # 'phase' or 'operation'
MODEL_NAME="${MODEL_NAME:-${5:-Qwen/Qwen2.5-7B-Instruct}}"
POLICY="${POLICY:-${6:-ppo}}"  # ppo | remax | grpo | sft
NNODES="${NNODES:-${7:-1}}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${8:-1}}"
DATASET_NAME="${DATASET_NAME:-${9:-gsm8k}}"
GPU_ID=0
POWER_CAP_W_RAW="${POWER_CAP_W:-}"
POWER_CAP_W_EFFECTIVE=""
POWER_CAP_APPLIED=0
declare -A POWER_CAP_ORIG_LIMITS

case "${POWER_CAP_W_RAW,,}" in
    ""|"0"|"default"|"none"|"null"|"off")
        POWER_CAP_W_EFFECTIVE=""
        ;;
    *)
        POWER_CAP_W_EFFECTIVE="$POWER_CAP_W_RAW"
        ;;
esac

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${TIMESTAMP}"

# -------------------- Paths --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PROFILING_DIR="${SCRIPT_DIR}"
export VERL_PROFILER_DIR="$PROFILING_DIR"
MONITOR_ROOT="${SCRATCH_DIR}/monitoring"
case "${USE_VALIDATION:-0}" in
    1|true|TRUE|yes|YES)
        MONITOR_ROOT="${SCRATCH_DIR}/monitoring_val"
        ;;
esac
MONITORING_DIR="${MONITOR_ROOT}/${EXPERIMENT_NAME}"

mkdir -p "$MONITORING_DIR"
cd "$PROJECT_DIR"

export VERL_FILE_LOGGER_ROOT="$MONITOR_ROOT"
export VERL_FILE_LOGGER_PATH="${MONITORING_DIR}/${EXPERIMENT_NAME}.jsonl"

export EXPERIMENT_NAME
export MONITORING_DIR
export PROJECT_DIR
export SCRATCH_DIR
export NNODES
export N_GPUS_PER_NODE
# Telemetry sampling controls for JSONL NVML/RAPL collector.
# Keep POLL_INTERVAL as the single source of truth.
export VERL_TELEMETRY_SAMPLE_INTERVAL_S="$POLL_INTERVAL"

# Persist experiment metadata alongside monitoring outputs (scratch + later migrated)
echo "$EXPERIMENT_NAME" > "${MONITORING_DIR}/experiment_name.txt"
if [ -n "${RUN_CONFIG_JSON:-}" ]; then
    python3 - <<'PY'
import json
import os

raw = os.environ.get("RUN_CONFIG_JSON", "")
out_path = os.environ.get("MONITORING_DIR", "") + "/run_config.json"
if raw and out_path:
    try:
        obj = json.loads(raw)
    except Exception:
        obj = {"raw": raw}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
PY
elif [ -n "${RUN_CONFIG_PATH:-}" ] && [ -f "${RUN_CONFIG_PATH:-}" ]; then
    cp "${RUN_CONFIG_PATH}" "${MONITORING_DIR}/run_config.json"
fi

if [ -n "${SLURM_CONFIG_PATH:-}" ] && [ -f "${SLURM_CONFIG_PATH:-}" ]; then
    cp "${SLURM_CONFIG_PATH}" "${MONITORING_DIR}/slurm_config.json"
fi

# Persist the resolved experiment name for Slurm cleanup/migration
if [ -n "${SLURM_JOB_ID:-}" ]; then
    EXPERIMENT_NAME_FILE="${SCRATCH_DIR}/logs/.experiment_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
    echo "$EXPERIMENT_NAME" > "$EXPERIMENT_NAME_FILE"
fi

echo "=========================================="
echo "verl Phase/Subphase Profiling"
echo "=========================================="
echo "Experiment (canonical): $EXPERIMENT_NAME"
echo "Epochs: $TOTAL_EPOCHS"
echo "GPU monitor index: $GPU_ID"
echo "Poll Interval: ${POLL_INTERVAL}s"
echo "Monitoring Dir: $MONITORING_DIR"
echo "Granularity: $GRANULARITY (phase | operation for subphase timings)"
echo "Model: $MODEL_NAME"
echo "Policy: $POLICY"
echo "Nodes: $NNODES (gpus per node: $N_GPUS_PER_NODE)"
echo "Dataset: $DATASET_NAME"
if [ -n "$POWER_CAP_W_EFFECTIVE" ]; then
    echo "Requested Power Cap: ${POWER_CAP_W_EFFECTIVE}W"
else
    echo "Requested Power Cap: default (no cap)"
fi
echo "=========================================="

# -------------------- Optional GPU Power Cap --------------------
if [ -n "$POWER_CAP_W_EFFECTIVE" ]; then
    if [[ ! "$POWER_CAP_W_EFFECTIVE" =~ ^[0-9]+$ ]]; then
        echo "WARNING: POWER_CAP_W must be an integer watts value; got '$POWER_CAP_W_EFFECTIVE'. Ignoring."
    elif ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "WARNING: nvidia-smi not found; cannot apply POWER_CAP_W."
    else
        readarray -t GPU_INDICES < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | xargs -n1 echo)
        if [ "${#GPU_INDICES[@]}" -eq 0 ]; then
            echo "WARNING: No GPUs visible to nvidia-smi; cannot apply POWER_CAP_W."
        else
            echo "Applying power cap ${POWER_CAP_W_EFFECTIVE}W to visible GPUs: ${GPU_INDICES[*]}"
            for IDX in "${GPU_INDICES[@]}"; do
                ORIG_LIMIT_RAW="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i "$IDX" 2>/dev/null | head -1)"
                ORIG_LIMIT="$(echo "$ORIG_LIMIT_RAW" | awk '{print int($1)}')"
                if [ -n "$ORIG_LIMIT" ]; then
                    POWER_CAP_ORIG_LIMITS["$IDX"]="$ORIG_LIMIT"
                fi

                CAP_SET_OUTPUT="$(nvidia-smi -pl "$POWER_CAP_W_EFFECTIVE" -i "$IDX" 2>&1)"
                if [ $? -eq 0 ]; then
                    POWER_CAP_APPLIED=1
                    VERIFY_LINE="$(nvidia-smi --query-gpu=power.limit,enforced.power.limit,clocks_throttle_reasons.sw_power_cap --format=csv,noheader -i "$IDX" 2>/dev/null | head -1)"
                    echo "Power-cap verification (gpu $IDX): $VERIFY_LINE"
                else
                    echo "WARNING: Failed to set power cap on gpu $IDX. nvidia-smi output: $CAP_SET_OUTPUT"
                fi
            done
            if [ "$POWER_CAP_APPLIED" -ne 1 ]; then
                echo "WARNING: POWER_CAP_W was requested but could not be applied on any GPU."
            fi
        fi
    fi
fi

# -------------------- Cleanup --------------------
cleanup() {
    echo ""
    echo "Cleaning up..."

    # If a stray local run dir exists, migrate it into monitoring
    if [ -d "${PROJECT_DIR}/${EXPERIMENT_NAME}" ]; then
        echo "Relocating local outputs from ${PROJECT_DIR}/${EXPERIMENT_NAME} to ${MONITORING_DIR}..."
        rsync -avz "${PROJECT_DIR}/${EXPERIMENT_NAME}/" "${MONITORING_DIR}/" || true
        rm -rf "${PROJECT_DIR:?}/${EXPERIMENT_NAME}"
    fi

    if [ "${RAY_MANAGED_BY_SLURM:-0}" = "1" ]; then
        echo "Stopping Ray..."
        ray stop 2>/dev/null || true
    fi

    if [ "$POWER_CAP_APPLIED" -eq 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
        echo "Restoring original GPU power limits..."
        for IDX in "${!POWER_CAP_ORIG_LIMITS[@]}"; do
            ORIG_LIMIT="${POWER_CAP_ORIG_LIMITS[$IDX]}"
            nvidia-smi -pl "$ORIG_LIMIT" -i "$IDX" >/dev/null 2>&1 || true
        done
    fi

    # Remove phase state file ONLY (CSV + JSONL are data)
    rm -f "${MONITORING_DIR}/phase_state_${EXPERIMENT_NAME}.json"

    # Best-effort cleanup of node-local caches to reduce stale handles
    rm -rf "$RAY_TMPDIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" 2>/dev/null || true

    echo "Cleanup complete."
}

trap cleanup EXIT INT TERM

# -------------------- Start Training --------------------
echo ""
echo "Starting training..."
echo ""

if [ -z "${TRAIN_SCRIPT:-}" ]; then
    TRAIN_SCRIPT="run_verl_train_nonval.sh"
    case "${USE_VALIDATION:-}" in
        1|true|TRUE|yes|YES)
            TRAIN_SCRIPT="run_verl_train_val.sh"
            ;;
    esac
fi

export PYTHONUNBUFFERED=1
export TRAIN_LOG_FILE="${SCRATCH_DIR}/logs/${EXPERIMENT_NAME}.log"

bash "${PROFILING_DIR}/${TRAIN_SCRIPT}" \
    "$EXPERIMENT_NAME" \
    "$TOTAL_EPOCHS" \
    "$GRANULARITY" \
    "$MODEL_NAME" \
    "$POLICY" \
    "$NNODES" \
    "$N_GPUS_PER_NODE" \
    "$DATASET_NAME"

echo ""
echo "Training complete."
