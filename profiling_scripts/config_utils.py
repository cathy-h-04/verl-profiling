#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shlex
import sys


RUN_KEYS = {
    "name",
    "model",
    "total_epochs",
    "poll_interval",
    "granularity",
    "policy",
    "nnodes",
    "gpus_per_node",
    "dataset",
    "use_validation",
    "val_freq",
    "val_max_samples",
    "total_steps",
    "save_freq",
    "power_cap_w",
    "resume_path",
    "rollout_n",
}

TRAIN_KEYS = {
    "tensor_parallel_size",
    "train_batch_size",
    "ppo_mini_batch_size",
    "micro_batch_size_per_gpu",
    "log_prob_micro_batch_size",
    "gpu_memory_util",
    "rollout_max_batched_tokens",
    "rollout_max_model_len",
    "rollout_max_num_seqs",
    "enable_chunked_prefill",
    "rollout_quantization",
    "enable_grad_checkpointing",
    "reward_model_enable",
    "reward_model_name",
    "reward_model_micro_batch_size_per_gpu",
}

ENV_MAP_RUN = {
    "name": "BASE_EXPERIMENT_NAME",
    "model": "MODEL_NAME",
    "total_epochs": "TOTAL_EPOCHS",
    "poll_interval": "POLL_INTERVAL",
    "granularity": "GRANULARITY",
    "policy": "POLICY",
    "nnodes": "NNODES",
    "gpus_per_node": "N_GPUS_PER_NODE",
    "dataset": "DATASET_NAME",
    "use_validation": "USE_VALIDATION",
    "val_freq": "VAL_FREQ",
    "val_max_samples": "VAL_MAX_SAMPLES",
    "total_steps": "TOTAL_STEPS",
    "save_freq": "SAVE_FREQ",
    "power_cap_w": "POWER_CAP_W",
    "resume_path": "RESUME_FROM_CHECKPOINT",
    "rollout_n": "ROLLOUT_N",
}

ENV_MAP_TRAIN = {
    "tensor_parallel_size": "TENSOR_PARALLEL_SIZE",
    "train_batch_size": "TRAIN_BATCH_SIZE",
    "ppo_mini_batch_size": "PPO_MINI_BATCH_SIZE",
    "micro_batch_size_per_gpu": "MICRO_BATCH_SIZE_PER_GPU",
    "log_prob_micro_batch_size": "LOG_PROB_MICRO_BATCH_SIZE",
    "gpu_memory_util": "GPU_MEMORY_UTIL",
    "rollout_max_batched_tokens": "ROLLOUT_MAX_BATCHED_TOKENS",
    "rollout_max_model_len": "ROLLOUT_MAX_MODEL_LEN",
    "rollout_max_num_seqs": "ROLLOUT_MAX_NUM_SEQS",
    "enable_chunked_prefill": "ROLLOUT_ENABLE_CHUNKED_PREFILL",
    "rollout_quantization": "ROLLOUT_QUANTIZATION",
    "enable_grad_checkpointing": "ENABLE_GRAD_CHECKPOINTING",
    "reward_model_enable": "REWARD_MODEL_ENABLE",
    "reward_model_name": "RM_MODEL_NAME",
    "reward_model_micro_batch_size_per_gpu": "RM_MICRO_BATCH_SIZE_PER_GPU",
}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_keys(src: dict, keys: set):
    return {k: src[k] for k in keys if k in src}


def _merge(a: dict, b: dict):
    out = dict(a)
    out.update(b)
    return out


def _collect_defaults(cfg: dict):
    defaults = cfg.get("defaults", {})
    run_defaults = {}
    train_defaults = {}

    if isinstance(defaults, dict):
        if "run" in defaults and isinstance(defaults["run"], dict):
            run_defaults = _merge(run_defaults, _extract_keys(defaults["run"], RUN_KEYS))
        if "train" in defaults and isinstance(defaults["train"], dict):
            train_defaults = _merge(train_defaults, _extract_keys(defaults["train"], TRAIN_KEYS))

        run_defaults = _merge(run_defaults, _extract_keys(defaults, RUN_KEYS))
        train_defaults = _merge(train_defaults, _extract_keys(defaults, TRAIN_KEYS))

    return run_defaults, train_defaults


def _get_runs(cfg: dict):
    if "runs" in cfg:
        runs = cfg["runs"]
        if not isinstance(runs, list):
            raise ValueError("config 'runs' must be a list")
        return runs
    if "run" in cfg:
        run = cfg["run"]
        if not isinstance(run, dict):
            raise ValueError("config 'run' must be an object")
        return [run]
    raise ValueError("config must contain 'runs' (list) or 'run' (object)")


def select_run(cfg: dict, index, name):
    runs = _get_runs(cfg)
    if name:
        for i, run in enumerate(runs, start=1):
            if isinstance(run, dict) and run.get("name") == name:
                return run, i, name
        raise ValueError(f"no run with name '{name}' found")
    if index is None:
        raise ValueError("either --index or --name is required")
    if index < 1 or index > len(runs):
        raise ValueError(f"index {index} out of range (1..{len(runs)})")
    run = runs[index - 1]
    if not isinstance(run, dict):
        raise ValueError("run entry must be an object")
    return run, index, run.get("name")


def resolve_run(cfg: dict, index, name):
    run_defaults, train_defaults = _collect_defaults(cfg)
    run, sel_index, sel_name = select_run(cfg, index, name)

    run_part = _extract_keys(run, RUN_KEYS)
    train_part = _extract_keys(run, TRAIN_KEYS)

    if "train" in run and isinstance(run["train"], dict):
        train_part = _merge(train_part, _extract_keys(run["train"], TRAIN_KEYS))

    resolved_run = _merge(run_defaults, run_part)
    resolved_train = _merge(train_defaults, train_part)

    if not resolved_run.get("name"):
        raise ValueError("run 'name' is required")

    meta = {
        "source": os.path.abspath(cfg["_config_path"]),
        "index": sel_index,
        "name": sel_name or resolved_run.get("name"),
    }
    return {
        "meta": meta,
        "run": resolved_run,
        "train": resolved_train,
    }


def to_env_value(value, *, bool_as_num=False):
    if value is None:
        return ""
    if isinstance(value, bool):
        if bool_as_num:
            return "1" if value else "0"
        return "true" if value else "false"
    return str(value)


def emit_exports(resolved: dict, config_path: str):
    env = {}
    for k, v in resolved.get("run", {}).items():
        if k in ENV_MAP_RUN:
            env_key = ENV_MAP_RUN[k]
            env_val = to_env_value(v, bool_as_num=(env_key == "USE_VALIDATION"))
            env[env_key] = env_val
    for k, v in resolved.get("train", {}).items():
        if k in ENV_MAP_TRAIN:
            env_key = ENV_MAP_TRAIN[k]
            env_val = to_env_value(v, bool_as_num=False)
            env[env_key] = env_val

    config_json = json.dumps(resolved, separators=(",", ":"), sort_keys=False)
    config_hash = hashlib.sha1(config_json.encode("utf-8")).hexdigest()

    env["RUN_CONFIG_JSON"] = config_json
    env["RUN_CONFIG_HASH"] = config_hash
    env["RUN_CONFIG_PATH"] = os.path.abspath(config_path)
    env["RUN_CONFIG_NAME"] = resolved.get("meta", {}).get("name") or ""
    env["RUN_CONFIG_INDEX"] = str(resolved.get("meta", {}).get("index") or "")

    for key in sorted(env.keys()):
        val = env[key]
        print(f"export {key}={shlex.quote(val)}")


def cmd_count(cfg: dict):
    runs = _get_runs(cfg)
    print(len(runs))


def cmd_max_gpus(cfg: dict):
    runs = _get_runs(cfg)
    run_defaults, _ = _collect_defaults(cfg)
    max_gpus = 0
    for run in runs:
        if not isinstance(run, dict):
            continue
        gpus = run.get("gpus_per_node")
        if gpus is None:
            gpus = run_defaults.get("gpus_per_node")
        try:
            gpus_val = int(gpus)
        except (TypeError, ValueError):
            gpus_val = 0
        if gpus_val > max_gpus:
            max_gpus = gpus_val
    print(max_gpus)


def cmd_resolve(cfg: dict, index, name, pretty: bool):
    resolved = resolve_run(cfg, index, name)
    if pretty:
        print(json.dumps(resolved, indent=2, sort_keys=False))
    else:
        print(json.dumps(resolved, separators=(",", ":"), sort_keys=False))


def cmd_export(cfg: dict, index, name, config_path: str):
    resolved = resolve_run(cfg, index, name)
    emit_exports(resolved, config_path)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run config utilities")
    sub = parser.add_subparsers(dest="cmd")

    p_count = sub.add_parser("count", help="print number of runs")
    p_count.add_argument("--config", required=True)

    p_max = sub.add_parser("max-gpus", help="print max gpus_per_node across runs")
    p_max.add_argument("--config", required=True)

    p_resolve = sub.add_parser("resolve", help="print resolved run config JSON")
    p_resolve.add_argument("--config", required=True)
    p_resolve.add_argument("--index", type=int)
    p_resolve.add_argument("--name")
    p_resolve.add_argument("--pretty", action="store_true")

    p_export = sub.add_parser("export", help="print shell exports for run")
    p_export.add_argument("--config", required=True)
    p_export.add_argument("--index", type=int)
    p_export.add_argument("--name")

    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        parser.exit(2, "ERROR: missing subcommand\n")
    return args


def main(argv):
    args = parse_args(argv)
    cfg = load_json(args.config)
    cfg["_config_path"] = args.config

    try:
        if args.cmd == "count":
            cmd_count(cfg)
        elif args.cmd == "max-gpus":
            cmd_max_gpus(cfg)
        elif args.cmd == "resolve":
            cmd_resolve(cfg, args.index, args.name, args.pretty)
        elif args.cmd == "export":
            cmd_export(cfg, args.index, args.name, args.config)
        else:
            raise ValueError(f"unknown command {args.cmd}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
