#!/usr/bin/env python3
import argparse
import json
import sys


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def emit_args(args):
    for arg in args:
        print(arg)


def sbatch_args(cfg: dict):
    args = []

    def add_kv(flag, value):
        if value is None or value == "":
            return
        args.append(f"{flag}={value}")

    def add_flag(flag, value):
        if value:
            args.append(flag)

    mapping = {
        "job_name": "--job-name",
        "nodes": "--nodes",
        "ntasks_per_node": "--ntasks-per-node",
        "mem": "--mem",
        "partition": "--partition",
        "time": "--time",
        "cpus_per_task": "--cpus-per-task",
        "output": "--output",
        "error": "--error",
        "account": "--account",
        "qos": "--qos",
        "constraint": "--constraint",
        "comment": "--comment",
        "signal": "--signal",
    }

    for key, flag in mapping.items():
        add_kv(flag, cfg.get(key))

    gres = cfg.get("gres")
    if gres:
        add_kv("--gres", gres)
    else:
        gpus = cfg.get("gpus_per_node")
        if gpus is not None and gpus != "":
            add_kv("--gres", f"gpu:{gpus}")

    add_flag("--exclusive", cfg.get("exclusive"))
    add_flag("--requeue", cfg.get("requeue"))

    extra = cfg.get("extra_args") or []
    if isinstance(extra, list):
        for item in extra:
            if isinstance(item, str) and item.strip():
                args.append(item.strip())

    return args


def mail_args(cfg: dict):
    args = []
    mail_user = cfg.get("mail_user")
    mail_type = cfg.get("mail_type")
    if mail_user:
        args.append(f"--mail-user={mail_user}")
    if mail_type:
        args.append(f"--mail-type={mail_type}")
    return args


def validate_config(cfg: dict):
    missing = []
    for key in ("output", "error"):
        value = cfg.get(key)
        if value is None or value == "":
            missing.append(key)
    if missing:
        raise ValueError(
            "slurm config is missing required field(s): " + ", ".join(missing) + ". "
            "Please set explicit log paths in slurm.json."
        )


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Slurm config utilities")
    sub = parser.add_subparsers(dest="cmd")

    p_sbatch = sub.add_parser("sbatch-args", help="print sbatch args")
    p_sbatch.add_argument("--config", required=True)

    p_mail = sub.add_parser("mail-args", help="print mail args")
    p_mail.add_argument("--config", required=True)

    p_validate = sub.add_parser("validate-config", help="validate required slurm config fields")
    p_validate.add_argument("--config", required=True)

    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        parser.exit(2, "ERROR: missing subcommand\n")
    return args


def main(argv):
    args = parse_args(argv)
    try:
        cfg = load_json(args.config)
        if args.cmd == "sbatch-args":
            emit_args(sbatch_args(cfg))
        elif args.cmd == "mail-args":
            emit_args(mail_args(cfg))
        elif args.cmd == "validate-config":
            validate_config(cfg)
        else:
            raise ValueError(f"unknown command {args.cmd}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
