#!/bin/bash
# ================================================================
# Verl PPO Training Script - Validation (Env-driven)
# Mirrors run_verl_train_nonval.sh with validation enabled.
# ================================================================

set -euo pipefail

# Ensure SCRATCH_DIR is set for the cluster
: "${SCRATCH_DIR:?ERROR: SCRATCH_DIR must be set to a shared writable scratch directory before launch.}"
export SCRATCH_DIR
mkdir -p "$SCRATCH_DIR/logs" "$SCRATCH_DIR/checkpoints" "$SCRATCH_DIR/data"
if [ -n "${RAY_ADDRESS:-}" ]; then
    export RAY_ADDRESS
fi
# Prefer node-local scratch for temp/cache to avoid NFS stale file handles.
LOCAL_SCRATCH="${LOCAL_SCRATCH:-${SLURM_TMPDIR:-/tmp/${USER}}}"
mkdir -p "$LOCAL_SCRATCH"
export TMPDIR="${TMPDIR:-${LOCAL_SCRATCH}/tmp}"
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LOCAL_SCRATCH}/xdg_cache}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-${LOCAL_SCRATCH}/vllm_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${LOCAL_SCRATCH}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_SCRATCH}/triton}"
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# -------------------- CLI Overrides --------------------
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume_path)
            RESUME_FROM_CHECKPOINT="$2"
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    export RESUME_FROM_CHECKPOINT
    echo "Resume from checkpoint: $RESUME_FROM_CHECKPOINT"
    if [ ! -d "$RESUME_FROM_CHECKPOINT" ]; then
        echo "FATAL: resume_path does not exist: $RESUME_FROM_CHECKPOINT"
        exit 1
    fi
    ACTOR_SEARCH_DIR="$RESUME_FROM_CHECKPOINT"
    if [ -d "$RESUME_FROM_CHECKPOINT/actor" ]; then
        ACTOR_SEARCH_DIR="$RESUME_FROM_CHECKPOINT/actor"
    fi
    ACTOR_FILE=$(find "$ACTOR_SEARCH_DIR" -type f -name "model_world_size_*_rank_0.pt" | head -1)
    if [ -z "$ACTOR_FILE" ]; then
        echo "FATAL: resume_path missing actor/policy weights: $RESUME_FROM_CHECKPOINT"
        exit 1
    fi
fi

# -------------------- Configuration --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
EXPERIMENT_NAME="${1:-gsm8k_val_profile}"
TOTAL_EPOCHS="${2:-1}"
GRANULARITY="${3:-phase}"  # 'phase' or 'operation'
MODEL_NAME="${4:-Qwen/Qwen2.5-7B-Instruct}"
POLICY="${5:-ppo}"  # ppo | remax
NNODES="${6:-1}"
N_GPUS_PER_NODE="${7:-4}"
DATASET_NAME="${8:-gsm8k}"
VAL_FREQ="${VAL_FREQ:-${9:-20}}"  # validation frequency in training steps
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-${10:-}}"
TOTAL_STEPS="${TOTAL_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-}"
export EXPERIMENT_NAME

# Rollout tensor parallel size
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
# Batch size configuration - SMALL for fast profiling
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-4}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.50}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-8192}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-2048}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-false}"
ROLLOUT_QUANTIZATION="${ROLLOUT_QUANTIZATION:-}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ENABLE_GRAD_CHECKPOINTING="${ENABLE_GRAD_CHECKPOINTING:-true}"
RM_MODEL_NAME="${RM_MODEL_NAME:-sfairXC/FsfairX-LLaMA3-RM-v0.1}"
RM_MICRO_BATCH_SIZE_PER_GPU="${RM_MICRO_BATCH_SIZE_PER_GPU:-16}"
REWARD_MODEL_ENABLE_RAW="${REWARD_MODEL_ENABLE:-}"

# -------------------- Environment Setup --------------------
cd "$PROJECT_DIR"

VENV_ACTIVATE="${VENV_ACTIVATE:-${PROJECT_DIR}/verl-env/bin/activate}"
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "ERROR: VENV_ACTIVATE not found: $VENV_ACTIVATE"
    exit 1
fi
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_DIR}/verl"
export PYTHONUNBUFFERED=1
# Optional token script for gated model access
TOKEN_SCRIPT="${SCRIPT_DIR}/token.sh"
if [ -f "$TOKEN_SCRIPT" ]; then
    if ! bash "$TOKEN_SCRIPT"; then
        echo "WARNING: token script failed: $TOKEN_SCRIPT (continuing)"
    fi
else
    echo "WARNING: token script not found: $TOKEN_SCRIPT (continuing)"
fi
# Structured file logging (JSONL) goes to $SCRATCH_DIR/monitoring_val/<experiment>.jsonl
MONITORING_DIR="${MONITORING_DIR:-${SCRATCH_DIR}/monitoring_val/${EXPERIMENT_NAME}}"
export VERL_FILE_LOGGER_ROOT="$(dirname "$MONITORING_DIR")"
export VERL_FILE_LOGGER_PATH="${MONITORING_DIR}/${EXPERIMENT_NAME}.jsonl"
mkdir -p "$MONITORING_DIR"

# Flash attention enabled (requires compatible flash-attn install)
# export VLLM_DISABLE_FLASHINFER=1  # uncomment if flashinfer causes issues

# -------------------- Dataset Setup --------------------
DATASET_ARGS=()
RM_ARGS=()
case "$DATASET_NAME" in
    gsm8k)
        DATA_DIR="${SCRATCH_DIR}/data/gsm8k"
        TRAIN_FILE="${DATA_DIR}/train.parquet"
        VAL_FILE="${DATA_DIR}/test.parquet"
        PREPROCESS_CMD="python3 examples/data_preprocess/gsm8k.py --local_save_dir \"$DATA_DIR\""
        ;;
    rlhf-ff)
        # RLHF full-hh-rlhf smoke path.
        RLHF_ROOT="${SCRATCH_DIR}/data/full_hh_rlhf"
        DATA_DIR="${RLHF_ROOT}/rl"
        TRAIN_FILE="${DATA_DIR}/train.parquet"
        VAL_FILE="${DATA_DIR}/test.parquet"
        PREPROCESS_CMD="python3 examples/data_preprocess/full_hh_rlhf.py --split rl --local_save_dir \"$RLHF_ROOT\""
        # Backward compatibility for older preprocessed dirs that only contain train.parquet.
        if [ ! -f "$VAL_FILE" ] && [ -f "$TRAIN_FILE" ]; then
            echo "WARNING: ${VAL_FILE} not found; falling back to ${TRAIN_FILE} for validation."
            VAL_FILE="$TRAIN_FILE"
        fi
        # full_hh_rlhf prompts can exceed 512 tokens.
        DATASET_ARGS+=("data.max_prompt_length=1024")
        DATASET_ARGS+=("data.truncation=right")
        # For full_hh_rlhf, repo examples use model-based RM.
        RM_ARGS+=("reward_model.enable=True")
        RM_ARGS+=("reward_model.model.path=${RM_MODEL_NAME}")
        RM_ARGS+=("reward_model.micro_batch_size_per_gpu=${RM_MICRO_BATCH_SIZE_PER_GPU}")
        ;;
    *)
        echo "ERROR: Unsupported dataset '$DATASET_NAME' (supported: gsm8k, rlhf-ff)"
        exit 1
        ;;
esac

# Optional config override to force-enable/disable model RM on any dataset.
# Backward compatible: if REWARD_MODEL_ENABLE is unset, dataset defaults above are preserved.
if [ -n "$REWARD_MODEL_ENABLE_RAW" ]; then
    case "${REWARD_MODEL_ENABLE_RAW,,}" in
        1|true|yes|y|on)
            RM_ARGS=(
                "reward_model.enable=True"
                "reward_model.model.path=${RM_MODEL_NAME}"
                "reward_model.micro_batch_size_per_gpu=${RM_MICRO_BATCH_SIZE_PER_GPU}"
            )
            echo "INFO: Enabling model RM via REWARD_MODEL_ENABLE override."
            ;;
        0|false|no|n|off)
            RM_ARGS=()
            echo "INFO: Disabling model RM via REWARD_MODEL_ENABLE override."
            ;;
        *)
            echo "ERROR: Invalid REWARD_MODEL_ENABLE='$REWARD_MODEL_ENABLE_RAW' (expected true/false)."
            exit 1
            ;;
    esac
fi

# -------------------- Directory Setup --------------------
OUTPUT_DIR="${SCRATCH_DIR}/checkpoints/${EXPERIMENT_NAME}"
LOG_DIR="${SCRATCH_DIR}/logs"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR"

# -------------------- System Check --------------------
echo "========================================"
echo "Verl PPO Training - Validation Mode"
echo "========================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Model: $MODEL_NAME"
echo "Epochs: $TOTAL_EPOCHS"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "Profiling Granularity: $GRANULARITY"
echo "Policy: $POLICY"
echo "Nodes: $NNODES (gpus per node: $N_GPUS_PER_NODE)"
echo "Dataset: $DATASET_NAME"
echo "Validation Frequency: $VAL_FREQ"
echo "Python: $(python --version)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "Verl Version: $(python -c 'import verl; print(verl.__version__)' 2>/dev/null || echo 'Not found')"
echo "========================================"
echo ""

# -------------------- Data Preparation --------------------
if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
    echo "Preparing dataset ($DATASET_NAME)..."
    eval "$PREPROCESS_CMD"
    echo "Dataset prepared"
else
    echo "Dataset already exists"
fi
echo ""

# -------------------- Training --------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${TRAIN_LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}.log}"

echo "========================================"
echo "Starting PPO Training"
echo "========================================"
echo "Logs will be saved to: $LOG_FILE"
echo ""

POLICY_ARGS=()
case "$POLICY" in
    ppo)
        ;;
    remax)
        POLICY_ARGS+=("algorithm.adv_estimator=remax")
        ;;
    grpo)
        POLICY_ARGS+=("algorithm.adv_estimator=grpo")
        POLICY_ARGS+=("algorithm.use_kl_in_reward=False")
        POLICY_ARGS+=("actor_rollout_ref.actor.use_kl_loss=False")
        ;;
    *)
        echo "ERROR: Unsupported policy '$POLICY' (use ppo, remax, or grpo)"
        exit 1
        ;;
esac

RESUME_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
    RESUME_ARGS+=("trainer.resume_mode=resume_path")
    RESUME_ARGS+=("trainer.resume_from_path=$RESUME_FROM_CHECKPOINT")
fi
SAVE_ARGS=()
if [ -n "${SAVE_FREQ:-}" ] && [ "$SAVE_FREQ" -gt 0 ]; then
    SAVE_ARGS+=("trainer.save_freq=$SAVE_FREQ")
fi
RAY_INIT_ARGS=()
if [ -n "${RAY_ADDRESS:-}" ]; then
    RAY_INIT_ARGS+=("ray_kwargs.ray_init.num_cpus=null")
fi
if [ -n "${RAY_ADDRESS:-}" ] && [ "${#RAY_INIT_ARGS[@]}" -eq 0 ]; then
    echo "FATAL: ray_kwargs.ray_init.num_cpus must be null when RAY_ADDRESS is set."
    exit 1
fi
ROLLOUT_QUANT_ARGS=()
case "${ROLLOUT_QUANTIZATION,,}" in
    ""|"none"|"null"|"default"|"off")
        ;;
    *)
        ROLLOUT_QUANT_ARGS+=("+actor_rollout_ref.rollout.quantization=${ROLLOUT_QUANTIZATION}")
        ;;
esac


python3 -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.dataloader_num_workers=4 \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path="$MODEL_NAME" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRAD_CHECKPOINTING \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_BATCHED_TOKENS \
  actor_rollout_ref.rollout.enable_chunked_prefill=$ROLLOUT_ENABLE_CHUNKED_PREFILL \
  actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN \
  actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
  "${ROLLOUT_QUANT_ARGS[@]}" \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  critic.model.path="$MODEL_NAME" \
  critic.optim.lr=1e-5 \
  critic.model.enable_gradient_checkpointing=$ENABLE_GRAD_CHECKPOINTING \
  critic.model.fsdp_config.model_dtype=bfloat16 \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=[console,file] \
  trainer.project_name="$EXPERIMENT_NAME" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  +trainer.enable_phase_profiling=True \
  +trainer.phase_profiling_granularity="$GRANULARITY" \
  trainer.val_before_train=False \
  trainer.test_freq=$VAL_FREQ \
  ${VAL_MAX_SAMPLES:+data.val_max_samples=$VAL_MAX_SAMPLES} \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=$NNODES \
  "${RAY_INIT_ARGS[@]}" \
  ray_kwargs.timeline_json_file="${MONITORING_DIR}/ray_timeline.json" \
  "${SAVE_ARGS[@]}" \
  trainer.total_epochs=$TOTAL_EPOCHS \
  ${TOTAL_STEPS:+trainer.total_training_steps=$TOTAL_STEPS} \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="$OUTPUT_DIR" \
  +critic.model.override_config.attn_implementation=flash_attention_2 \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  "${RESUME_ARGS[@]}" \
  "${DATASET_ARGS[@]}" \
  "${RM_ARGS[@]}" \
  "${POLICY_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?
sleep 10

# -------------------- Post-Training --------------------
echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed (exit code: $EXIT_CODE)."
fi
echo "========================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Checkpoints: $OUTPUT_DIR"
echo "Logs: $LOG_FILE"
echo "========================================"
echo ""

echo "Final GPU Memory Usage:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | head -1
echo ""

exit $EXIT_CODE
