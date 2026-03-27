#!/usr/bin/env python3
"""Unified pre-run validator for Slurm preflight + mirrored veRL config checks."""

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
from typing import Optional


def _warn(msg: str):
    print(msg, file=sys.stderr)


def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if val is None or val == "" else val


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _extract_int(raw: str) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    m = re.search(r"(-?\d+)$", s)
    return int(m.group(1)) if m else None


def _env_int(name: str, default: int) -> int:
    val = _extract_int(os.environ.get(name))
    return default if val is None else val


def _env_opt_int(name: str):
    return _extract_int(os.environ.get(name))


def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str, param: str):
    per_gpu = f"{param}_per_gpu"
    if mbs is None and mbs_per_gpu is None:
        raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{per_gpu}'.")
    if mbs is not None and mbs_per_gpu is not None:
        raise ValueError(
            f"[{name}] You set both '{name}.{param}' and '{name}.{per_gpu}'. Remove '{name}.{param}' (deprecated)."
        )


def _max_gpus_from_runs(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    defaults = cfg.get("defaults", {}) if isinstance(cfg, dict) else {}
    run_defaults = {}
    if isinstance(defaults, dict):
        if isinstance(defaults.get("run"), dict):
            run_defaults.update(defaults["run"])
        run_defaults.update(defaults)

    runs = cfg.get("runs", [])
    if isinstance(cfg.get("run"), dict):
        runs = [cfg["run"]]
    if not isinstance(runs, list):
        raise ValueError("config 'runs' must be a list or 'run' must be an object")

    max_gpus = 0
    for run in runs:
        if not isinstance(run, dict):
            continue
        gpus = run.get("gpus_per_node", run_defaults.get("gpus_per_node"))
        g = _extract_int(str(gpus)) if gpus is not None else None
        if g is not None and g > max_gpus:
            max_gpus = g
    return max_gpus


def _emit_exports(values: dict):
    for key in sorted(values):
        print(f"export {key}={shlex.quote(str(values[key]))}")


def _run_checks() -> dict:
    resolved = {}

    runs_file = _env_str("RUNS_FILE", "")
    if not runs_file:
        raise ValueError("RUNS_FILE is required for pre-run validation.")
    if not os.path.isfile(runs_file):
        raise ValueError(f"RUNS_FILE not found: {runs_file}")

    requested_gpus = _extract_int(_env_str("REQUESTED_GPUS_PER_NODE", ""))
    if requested_gpus is None:
        _warn("WARNING: REQUESTED_GPUS_PER_NODE invalid; defaulting to 1.")
        requested_gpus = 1
    resolved["REQUESTED_GPUS_PER_NODE"] = requested_gpus

    max_param_gpus = _max_gpus_from_runs(runs_file)
    resolved["MAX_PARAM_GPUS"] = max_param_gpus
    if requested_gpus < max_param_gpus:
        raise ValueError(
            f"Requested GPUs per node ({requested_gpus}) is less than max GPUs in RUNS_FILE ({max_param_gpus})."
        )

    run_gpus_per_node = _extract_int(os.environ.get("N_GPUS_PER_NODE"))
    if run_gpus_per_node is None:
        run_gpus_per_node = requested_gpus
    resolved["RUN_GPUS_PER_NODE"] = run_gpus_per_node

    if requested_gpus > run_gpus_per_node:
        _warn(
            f"WARNING: Slurm requested {requested_gpus} GPU(s) but run requests only {run_gpus_per_node} GPU(s). "
            "This will leave GPUs idle."
        )
        if _env_bool("STRICT_GPU_MATCH", False):
            raise ValueError("STRICT_GPU_MATCH=1 and GPU mismatch detected.")

    ray_cpu_per_gpu_bundle = _env_int("RAY_CPU_PER_GPU_BUNDLE", 3)
    resolved["RAY_CPU_PER_GPU_BUNDLE"] = ray_cpu_per_gpu_bundle

    slurm_cpus = _extract_int(os.environ.get("SLURM_CPUS_PER_TASK"))
    if slurm_cpus is None:
        slurm_cpus = os.cpu_count() or 1
        _warn(f"WARNING: SLURM_CPUS_PER_TASK is not set; using nproc={slurm_cpus} for preflight checks.")
    resolved["SLURM_CPUS_PER_TASK_EFFECTIVE"] = slurm_cpus

    required_cpus = run_gpus_per_node * ray_cpu_per_gpu_bundle
    resolved["REQUIRED_CPUS_PER_NODE"] = required_cpus
    if slurm_cpus < required_cpus:
        raise ValueError(
            "Unschedulable Ray resources: "
            f"gpus_per_node={run_gpus_per_node}, RAY_CPU_PER_GPU_BUNDLE={ray_cpu_per_gpu_bundle}, "
            f"required_cpus={required_cpus}, provided_cpus={slurm_cpus}."
        )

    rollout_max_batched_tokens = _env_int("ROLLOUT_MAX_BATCHED_TOKENS", 8192)
    rollout_max_model_len = _env_int("ROLLOUT_MAX_MODEL_LEN", 2048)
    rollout_enable_chunked_prefill = _env_bool("ROLLOUT_ENABLE_CHUNKED_PREFILL", False)
    if rollout_max_batched_tokens < rollout_max_model_len and not rollout_enable_chunked_prefill:
        _warn(
            f"WARNING: rollout_max_batched_tokens ({rollout_max_batched_tokens}) < "
            f"rollout_max_model_len ({rollout_max_model_len})."
        )
        _warn("WARNING: Auto-enabling rollout chunked prefill for this run.")
        rollout_enable_chunked_prefill = True
    resolved["ROLLOUT_MAX_BATCHED_TOKENS"] = rollout_max_batched_tokens
    resolved["ROLLOUT_MAX_MODEL_LEN"] = rollout_max_model_len
    resolved["ROLLOUT_ENABLE_CHUNKED_PREFILL"] = "true" if rollout_enable_chunked_prefill else "false"

    run_config_json = os.environ.get("RUN_CONFIG_JSON", "")
    if run_config_json:
        obj = json.loads(run_config_json)
        train = obj.setdefault("train", {})
        train["rollout_max_batched_tokens"] = rollout_max_batched_tokens
        train["rollout_max_model_len"] = rollout_max_model_len
        train["enable_chunked_prefill"] = rollout_enable_chunked_prefill
        run_config_json = json.dumps(obj, separators=(",", ":"), sort_keys=False)
        resolved["RUN_CONFIG_JSON"] = run_config_json
        resolved["RUN_CONFIG_HASH"] = hashlib.sha1(run_config_json.encode("utf-8")).hexdigest()

    policy = _env_str("POLICY", "ppo").strip().lower()
    supported_policies = {"ppo", "remax", "grpo", "sft"}
    if policy not in supported_policies:
        raise ValueError(
            f"Unsupported POLICY='{policy}'. Supported values: {', '.join(sorted(supported_policies))}."
        )
    if policy == "sft":
        _warn("[preflight_validate_config] POLICY=sft detected; skipping PPO-specific config checks.")
        _warn("[preflight_validate_config] All checks passed.")
        return resolved

    n_gpus = _env_int("NNODES", 1) * _env_int("N_GPUS_PER_NODE", 4)
    train_batch_size = _env_int("TRAIN_BATCH_SIZE", 128)
    rollout_n = _env_int("ROLLOUT_N", 4)

    actor_use_dynamic_bsz = _env_bool("ACTOR_USE_DYNAMIC_BSZ", False)
    actor_ppo_mini_batch_size = _env_int("PPO_MINI_BATCH_SIZE", 32)
    actor_ppo_micro_batch_size = _env_opt_int("ACTOR_PPO_MICRO_BATCH_SIZE")
    actor_sp_size = _env_int("ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE", 1)
    actor_use_remove_padding = _env_bool("ACTOR_USE_REMOVE_PADDING", False)
    actor_use_kl_loss = _env_bool("ACTOR_USE_KL_LOSS", policy != "grpo")

    critic_enable_raw = os.environ.get("CRITIC_ENABLE")
    if critic_enable_raw is None or critic_enable_raw == "":
        use_critic = policy == "ppo"
    else:
        use_critic = critic_enable_raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    critic_use_dynamic_bsz = _env_bool("CRITIC_USE_DYNAMIC_BSZ", False)
    critic_ppo_mini_batch_size = _env_int("CRITIC_PPO_MINI_BATCH_SIZE", actor_ppo_mini_batch_size)
    critic_ppo_micro_batch_size = _env_opt_int("CRITIC_PPO_MICRO_BATCH_SIZE")
    critic_ppo_micro_batch_size_per_gpu = _env_opt_int("CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU")
    if critic_ppo_micro_batch_size_per_gpu is None:
        critic_ppo_micro_batch_size_per_gpu = _env_opt_int("MICRO_BATCH_SIZE_PER_GPU")
    critic_sp_size = _env_int("CRITIC_ULYSSES_SEQUENCE_PARALLEL_SIZE", 1)
    critic_use_remove_padding = _env_bool("CRITIC_USE_REMOVE_PADDING", False)

    reward_model_enable = _env_bool("REWARD_MODEL_ENABLE", _env_str("DATASET_NAME", "gsm8k").lower() == "rlhf-ff")
    reward_model_use_dynamic_bsz = _env_bool("REWARD_MODEL_USE_DYNAMIC_BSZ", False)
    reward_model_micro_batch_size = _env_opt_int("REWARD_MODEL_MICRO_BATCH_SIZE")
    reward_model_micro_batch_size_per_gpu = _env_opt_int("RM_MICRO_BATCH_SIZE_PER_GPU")
    if reward_model_micro_batch_size_per_gpu is None:
        reward_model_micro_batch_size_per_gpu = _env_opt_int("REWARD_MODEL_MICRO_BATCH_SIZE_PER_GPU")
    # Match runtime training scripts, which default RM micro-batch to 16.
    if reward_model_enable and reward_model_micro_batch_size is None and reward_model_micro_batch_size_per_gpu is None:
        reward_model_micro_batch_size_per_gpu = 16

    ref_log_prob_micro_batch_size = _env_opt_int("REF_LOG_PROB_MICRO_BATCH_SIZE")
    ref_log_prob_micro_batch_size_per_gpu = _env_opt_int("MICRO_BATCH_SIZE_PER_GPU")
    rollout_log_prob_micro_batch_size = _env_opt_int("ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE")
    rollout_log_prob_micro_batch_size_per_gpu = _env_opt_int("LOG_PROB_MICRO_BATCH_SIZE")

    algorithm_use_kl_in_reward = _env_bool("ALGORITHM_USE_KL_IN_REWARD", policy != "grpo")
    rollout_name = _env_str("ROLLOUT_NAME", "vllm").lower()
    lora_rank = _env_int("LORA_RANK", 0)
    val_do_sample = _env_bool("VAL_DO_SAMPLE", False)
    rollout_temperature = float(_env_str("ROLLOUT_TEMPERATURE", "1.0"))

    if not actor_use_dynamic_bsz:
        minimal_bsz = n_gpus
        real_train_batch_size = train_batch_size * rollout_n
        if real_train_batch_size % minimal_bsz != 0:
            raise ValueError(
                f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal batch size ({minimal_bsz})"
            )

    if not actor_use_dynamic_bsz:
        _check_mutually_exclusive(
            ref_log_prob_micro_batch_size,
            ref_log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.ref",
            "log_prob_micro_batch_size",
        )
        _check_mutually_exclusive(
            rollout_log_prob_micro_batch_size,
            rollout_log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
            "log_prob_micro_batch_size",
        )

    if reward_model_enable and not reward_model_use_dynamic_bsz:
        _check_mutually_exclusive(
            reward_model_micro_batch_size,
            reward_model_micro_batch_size_per_gpu,
            "reward_model",
            "micro_batch_size",
        )

    if not actor_use_dynamic_bsz:
        if train_batch_size < actor_ppo_mini_batch_size:
            raise ValueError(
                f"train_batch_size ({train_batch_size}) must be >= actor.ppo_mini_batch_size ({actor_ppo_mini_batch_size})"
            )
        if actor_ppo_micro_batch_size is not None:
            if actor_ppo_mini_batch_size % actor_ppo_micro_batch_size != 0:
                raise ValueError(
                    f"actor.ppo_mini_batch_size ({actor_ppo_mini_batch_size}) must be divisible by "
                    f"actor.ppo_micro_batch_size ({actor_ppo_micro_batch_size})"
                )
            if actor_ppo_micro_batch_size * actor_sp_size < n_gpus:
                raise ValueError(
                    f"actor.ppo_micro_batch_size ({actor_ppo_micro_batch_size}) * actor_sp_size ({actor_sp_size}) "
                    f"must be >= n_gpus ({n_gpus})"
                )
    if actor_sp_size > 1 and not actor_use_remove_padding:
        raise ValueError("When using actor sequence parallelism, enable actor use_remove_padding.")

    if use_critic:
        if not critic_use_dynamic_bsz:
            _check_mutually_exclusive(
                critic_ppo_micro_batch_size,
                critic_ppo_micro_batch_size_per_gpu,
                "critic",
                "micro_batch_size",
            )
            if critic_ppo_micro_batch_size is not None and critic_ppo_mini_batch_size % critic_ppo_micro_batch_size != 0:
                raise ValueError(
                    f"critic.ppo_mini_batch_size ({critic_ppo_mini_batch_size}) must be divisible by "
                    f"critic.ppo_micro_batch_size ({critic_ppo_micro_batch_size})"
                )
            if train_batch_size < critic_ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= critic.ppo_mini_batch_size ({critic_ppo_mini_batch_size})"
                )
            if critic_ppo_micro_batch_size is not None and critic_ppo_micro_batch_size * critic_sp_size < n_gpus:
                raise ValueError(
                    f"critic.ppo_micro_batch_size ({critic_ppo_micro_batch_size}) * critic_sp_size ({critic_sp_size}) "
                    f"must be >= n_gpus ({n_gpus})"
                )
        if critic_sp_size > 1 and not critic_use_remove_padding:
            raise ValueError("When using critic sequence parallelism, enable critic use_remove_padding.")

    if val_do_sample and rollout_temperature <= 0:
        raise ValueError("validation generation temperature must be > 0 when do_sample is true.")

    if lora_rank > 0 and rollout_name == "vllm" and lora_rank > 512:
        raise ValueError("LoRA rank in vLLM must be <= 512.")

    if algorithm_use_kl_in_reward and actor_use_kl_loss:
        _warn("NOTICE: Both in-reward KL and KL loss are enabled.")

    _warn("[preflight_validate_config] All checks passed.")
    return resolved


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Unified pre-run validator for profiling Slurm launches.")
    parser.add_argument("--emit-exports", action="store_true", help="Print shell export lines to stdout.")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    try:
        resolved = _run_checks()
    except Exception as exc:
        print(f"ERROR: pre-run validation failed: {exc}", file=sys.stderr)
        return 1

    if args.emit_exports:
        _emit_exports(resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
