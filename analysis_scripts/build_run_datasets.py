#!/usr/bin/env python3
"""Build canonical Parquet datasets from experiment run artifacts.

This script scans a results root for run folders that contain `experiment_name.txt`,
validates required artifacts, excludes incomplete runs, and writes normalized tables.

Default output: /n/home08/chou/verl_research/DATASETS
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from ingestion_checks import (
    build_phase_outlier_flags,
    build_boundary_integrity_outputs,
    build_step_index_and_checks,
    validate_ingestion_checks,
)
from periodic_aggregations import (
    add_time_weights,
    aggregate_periodic_metrics,
    filter_periodic_to_phase_window,
)
from utils import (
    _build_analysis_views,
    _build_phase_fact,
    _canonicalize_step,
    _classify_metric,
    _coalesce_join_column,
    _discover_run_dirs,
    _ensure_columns,
    _extract_lineage,
    _json_default,
    _parse_slurm_job_ids,
    _read_json,
    _read_jsonl,
    _read_text,
    _required_files_for_run,
    _safe_float,
    _safe_int,
    _sanitize_col,
    _stable_hash,
)


REQUIRED_STATIC_FILES = [
    "experiment_name.txt",
    "slurm_job_ids.txt",
    "slurm_config.json",
    "run_config.json",
    "nvml_boundary.jsonl",
    "nvml_periodic.jsonl",
    "rapl_boundary.jsonl",
    "rapl_periodic.jsonl",
    "tokens_and_steps.jsonl",
]

CRITICAL_JSONL_FILES = [
    "nvml_boundary.jsonl",
    "nvml_periodic.jsonl",
    "rapl_boundary.jsonl",
    "rapl_periodic.jsonl",
    "tokens_and_steps.jsonl",
]

# Curated convenience view keys.
CURATED_METRIC_KEYS = [
    "training/global_step",
    "training/epoch",
    "logging/validation_logged",
    "val-core/openai/gsm8k/reward/mean@1",
    "timing_s/step",
    "perf/throughput",
    "perf/total_num_tokens",
    "comm_s/step",
    "comm_s/update_actor",
    "actor/pg_loss",
    "actor/ppo_kl",
    "actor/entropy",
    "actor/lr",
    "critic/rewards/mean",
    "critic/advantages/mean",
    "critic/returns/mean",
    "response_length/mean",
    "prompt_length/mean",
    "rollout/straggler_ratio",
    "rollout/sync_efficiency",
    "perf/mfu/actor",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    "perf/cpu_memory_used_gb",
    "logging/wall_time",
]

TABLE_FILES = {
    "runs": "runs.parquet",
    "run_lineage": "run_lineage.parquet",
    "step_index_map": "step_index_map.parquet",
    "ingestion_checks": "ingestion_checks.parquet",
    "phase_instances": "phase_instances.parquet",
    "boundary_pair_integrity": "boundary_pair_integrity.parquet",
    "phase_fact": "phase_fact.parquet",
    "step_metrics_long": "step_metrics_long.parquet",
    "step_metrics_wide_curated": "step_metrics_wide_curated.parquet",
    "phase_timings_long": "phase_timings_long.parquet",
    "tokens_and_steps": "tokens_and_steps.parquet",
    "hardware_boundary": "hardware_boundary.parquet",
    "hardware_periodic": "hardware_periodic.parquet",
    "phase_summary": "phase_summary.parquet",
    "ingestion_report": "ingestion_report.parquet",
    "phase_fact_view": "phase_fact_view.parquet",
    "step_fact_view": "step_fact_view.parquet",
    "run_summary_view": "run_summary_view.parquet",
    "comparison_view": "comparison_view.parquet",
    "device_timeseries_view": "device_timeseries_view.parquet",
    "integrity_view": "integrity_view.parquet",
}


def _expected_gpu_count_for_run(run_id: str, slurm_config: Dict[str, Any]) -> Optional[int]:
    configured = _safe_int(slurm_config.get("gpus_per_node")) if isinstance(slurm_config, dict) else None
    if configured is not None and configured > 0:
        return configured
    rid = str(run_id).lower()
    if "2gpu_" in rid:
        return 2
    if "4gpu_" in rid:
        return 4
    return None


def _nvml_device_key(rec: Dict[str, Any]) -> Optional[str]:
    gpu_uuid = rec.get("gpu_uuid")
    if gpu_uuid not in (None, ""):
        return str(gpu_uuid)
    gpu_index = rec.get("gpu_index")
    if gpu_index is not None:
        return f"gpu_index:{gpu_index}"
    return None


def _infer_active_nvml_device_keys(
    nvml_periodic_records: List[Dict[str, Any]],
    expected_gpu_count: Optional[int],
) -> Optional[set[str]]:
    """Infer active GPU devices when extra idle GPUs are also polled.

    The current monitoring setup may poll all 4 local GPUs even for 2-GPU jobs.
    We rank devices by mean power first, then mean utilization, and keep the
    expected number of active GPUs for the run.
    """
    if expected_gpu_count is None or expected_gpu_count <= 0:
        return None

    device_rows: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for rec in nvml_periodic_records:
        key = _nvml_device_key(rec)
        if key is None:
            continue
        power_mw = _safe_float(rec.get("gpu_power_mW"))
        util_pct = _safe_float(rec.get("gpu_util_pct"))
        device_rows[key].append(
            (
                float(power_mw) if power_mw is not None else float("-inf"),
                float(util_pct) if util_pct is not None else float("-inf"),
            )
        )

    if not device_rows:
        return None
    if len(device_rows) <= expected_gpu_count:
        return set(device_rows.keys())

    ranked_rows = []
    for key, vals in device_rows.items():
        power_vals = [p for p, _ in vals if pd.notna(p) and p != float("-inf")]
        util_vals = [u for _, u in vals if pd.notna(u) and u != float("-inf")]
        mean_power = sum(power_vals) / len(power_vals) if power_vals else float("-inf")
        mean_util = sum(util_vals) / len(util_vals) if util_vals else float("-inf")
        ranked_rows.append((key, mean_power, mean_util))

    ranked_rows.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
    return {key for key, _, _ in ranked_rows[:expected_gpu_count]}

@dataclass
class RunQuality:
    run_dir: str
    run_id: Optional[str]
    status: str
    reason: str
    missing_files: List[str]
    zero_line_files: List[str]
    parse_error_files: Dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build run datasets from experiment artifacts.")
    parser.add_argument(
        "--results-root",
        default="/n/home08/chou/verl_research/results",
        help="Root directory containing experiment result folders.",
    )
    parser.add_argument(
        "--output-root",
        default="/n/home08/chou/verl_research/DATASETS",
        help="Directory where output Parquet tables will be written.",
    )
    parser.add_argument(
        "--include-subdir",
        action="append",
        default=None,
        help=(
            "Optional subdirectory under --results-root to include. "
            "Repeat this flag to include multiple subtrees. "
            "Accepts absolute paths or paths relative to --results-root."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Reserved for future parallel parsing. Current implementation is single-process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, remove existing output directory first.",
    )
    return parser.parse_args()


def _resolve_include_subdirs(results_root: Path, include_subdirs: Optional[List[str]]) -> List[Path]:
    if not include_subdirs:
        return []

    resolved_paths: List[Path] = []
    for raw_path in include_subdirs:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = results_root / candidate
        candidate = candidate.resolve()

        if not candidate.exists() or not candidate.is_dir():
            raise ValueError(f"--include-subdir path does not exist or is not a directory: {candidate}")

        try:
            candidate.relative_to(results_root)
        except ValueError as e:
            raise ValueError(
                f"--include-subdir path must be within --results-root ({results_root}): {candidate}"
            ) from e

        resolved_paths.append(candidate)

    return sorted(set(resolved_paths))


def build_datasets(results_root: Path, output_root: Path, overwrite: bool, include_dirs: Optional[List[Path]] = None) -> None:
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_run_dirs(results_root, include_dirs=include_dirs)

    runs_rows: List[Dict[str, Any]] = []
    lineage_rows: List[Dict[str, Any]] = []
    step_index_rows: List[Dict[str, Any]] = []
    ingestion_checks_rows: List[Dict[str, Any]] = []
    phase_instance_rows: List[Dict[str, Any]] = []
    boundary_pair_rows: List[Dict[str, Any]] = []
    step_metrics_long_rows: List[Dict[str, Any]] = []
    phase_timings_rows: List[Dict[str, Any]] = []
    tokens_rows: List[Dict[str, Any]] = []
    hw_boundary_rows: List[Dict[str, Any]] = []
    hw_periodic_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    wide_rows_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    validation_steps_by_run: Dict[str, set[int]] = defaultdict(set)
    curated_metric_cols = {_k: _sanitize_col(_k) for _k in CURATED_METRIC_KEYS}

    for run_dir in run_dirs:
        run_id: Optional[str] = None
        quality = RunQuality(
            run_dir=str(run_dir),
            run_id=None,
            status="excluded",
            reason="unknown",
            missing_files=[],
            zero_line_files=[],
            parse_error_files={},
        )

        try:
            run_id = _read_text(run_dir / "experiment_name.txt")
            quality.run_id = run_id

            required_files = _required_files_for_run(run_id, REQUIRED_STATIC_FILES)
            missing = [f for f in required_files if not (run_dir / f).exists()]
            if missing:
                quality.reason = "missing_required_files"
                quality.missing_files = missing
                report_rows.append(
                    {
                        **quality.__dict__,
                        "included": False,
                    }
                )
                continue

            exp_metrics_file = f"{run_id}.jsonl"
            phase_timings_file = f"phase_timings_{run_id}.jsonl"
            full_config_file = f"{run_id}_config.json"

            critical_files = CRITICAL_JSONL_FILES + [exp_metrics_file, phase_timings_file]
            nonempty_counts: Dict[str, int] = {}
            for fname in critical_files:
                read_res = _read_jsonl(run_dir / fname)
                nonempty_counts[fname] = read_res.nonempty_lines
                if read_res.nonempty_lines == 0:
                    quality.zero_line_files.append(fname)

            if quality.zero_line_files:
                quality.reason = "incomplete_zero_line_critical_files"
                report_rows.append(
                    {
                        **quality.__dict__,
                        "included": False,
                    }
                )
                continue

            run_config = _read_json(run_dir / "run_config.json")
            slurm_config = _read_json(run_dir / "slurm_config.json")
            full_config = _read_json(run_dir / full_config_file)
            slurm_ids = _parse_slurm_job_ids(run_dir / "slurm_job_ids.txt")

            logical_run_group = (
                run_config.get("run", {}).get("name") if isinstance(run_config.get("run"), dict) else None
            )
            resume_path = run_config.get("run", {}).get("resume_path") if isinstance(run_config.get("run"), dict) else None
            parent_run_name, resume_from_global_step = _extract_lineage(resume_path)

            # Parse core JSONL files once for included runs.
            read_map = {
                "metrics": _read_jsonl(run_dir / exp_metrics_file),
                "phase_timings": _read_jsonl(run_dir / phase_timings_file),
                "tokens": _read_jsonl(run_dir / "tokens_and_steps.jsonl"),
                "nvml_boundary": _read_jsonl(run_dir / "nvml_boundary.jsonl"),
                "nvml_periodic": _read_jsonl(run_dir / "nvml_periodic.jsonl"),
                "rapl_boundary": _read_jsonl(run_dir / "rapl_boundary.jsonl"),
                "rapl_periodic": _read_jsonl(run_dir / "rapl_periodic.jsonl"),
            }

            expected_gpu_count = _expected_gpu_count_for_run(run_id, slurm_config)
            active_nvml_device_keys = _infer_active_nvml_device_keys(
                read_map["nvml_periodic"].records,
                expected_gpu_count=expected_gpu_count,
            )

            for key, val in read_map.items():
                if val.parse_errors > 0:
                    quality.parse_error_files[key] = val.parse_errors

            if quality.parse_error_files:
                quality.reason = "json_parse_errors"
                report_rows.append(
                    {
                        **quality.__dict__,
                        "included": False,
                    }
                )
                continue

            # Included run.
            quality.status = "included"
            quality.reason = "ok"

            run_step_observations: List[Dict[str, Any]] = []
            run_boundary_rows_local: List[Dict[str, Any]] = []

            def register_step_observation(
                raw_step: Optional[int], raw_iteration: Optional[int], source_table: str
            ) -> Tuple[Optional[int], bool]:
                canonical_step, mismatch_flag = _canonicalize_step(raw_step, raw_iteration)
                run_step_observations.append(
                    {
                        "run_id": run_id,
                        "source_table": source_table,
                        "raw_step": raw_step,
                        "raw_iteration": raw_iteration,
                        "canonical_step": canonical_step,
                        "mismatch_flag": mismatch_flag,
                    }
                )
                return canonical_step, mismatch_flag

            configured_total_steps = (
                run_config.get("run", {}).get("total_steps") if isinstance(run_config.get("run"), dict) else None
            )
            run_row = {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "results_root": str(results_root),
                "logical_run_group": logical_run_group,
                "run_name": run_config.get("run", {}).get("name") if isinstance(run_config.get("run"), dict) else None,
                "model": run_config.get("run", {}).get("model") if isinstance(run_config.get("run"), dict) else None,
                "dataset": run_config.get("run", {}).get("dataset") if isinstance(run_config.get("run"), dict) else None,
                "policy": run_config.get("run", {}).get("policy") if isinstance(run_config.get("run"), dict) else None,
                "use_validation": run_config.get("run", {}).get("use_validation")
                if isinstance(run_config.get("run"), dict)
                else None,
                "val_freq": run_config.get("run", {}).get("val_freq") if isinstance(run_config.get("run"), dict) else None,
                "poll_interval": run_config.get("run", {}).get("poll_interval")
                if isinstance(run_config.get("run"), dict)
                else None,
                "total_steps": configured_total_steps,
                "configured_total_steps": configured_total_steps,
                "observed_total_steps": None,
                "total_steps_from_observed_fallback": False,
                "total_epochs": run_config.get("run", {}).get("total_epochs")
                if isinstance(run_config.get("run"), dict)
                else None,
                "resume_path": resume_path,
                "is_resumed_run": bool(resume_path),
                "resume_parent_run_name": parent_run_name,
                "resume_from_global_step": resume_from_global_step,
                "meta_source": run_config.get("meta", {}).get("source") if isinstance(run_config.get("meta"), dict) else None,
                "meta_index": run_config.get("meta", {}).get("index") if isinstance(run_config.get("meta"), dict) else None,
                "meta_name": run_config.get("meta", {}).get("name") if isinstance(run_config.get("meta"), dict) else None,
                "slurm_job_id": slurm_ids.get("slurm_job_id"),
                "slurm_array_task_id": slurm_ids.get("slurm_array_task_id"),
                "slurm_job_name": slurm_ids.get("slurm_job_name"),
                "slurm_timestamp": slurm_ids.get("timestamp"),
                "slurm_partition": slurm_config.get("partition"),
                "slurm_nodes": slurm_config.get("nodes"),
                "slurm_gpus_per_node": slurm_config.get("gpus_per_node"),
                "expected_gpu_count": expected_gpu_count,
                "active_nvml_device_count": (len(active_nvml_device_keys) if active_nvml_device_keys is not None else None),
                "active_nvml_device_ids_json": (
                    json.dumps(sorted(active_nvml_device_keys), ensure_ascii=True)
                    if active_nvml_device_keys is not None
                    else None
                ),
                "slurm_cpus_per_task": slurm_config.get("cpus_per_task"),
                "slurm_mem": slurm_config.get("mem"),
                "run_config_json": json.dumps(run_config, ensure_ascii=True, default=_json_default),
                "slurm_config_json": json.dumps(slurm_config, ensure_ascii=True, default=_json_default),
                "full_config_json": json.dumps(full_config, ensure_ascii=True, default=_json_default),
            }

            lineage_rows.append(
                {
                    "run_id": run_id,
                    "is_resumed_run": bool(resume_path),
                    "resume_path": resume_path,
                    "resume_parent_run_name": parent_run_name,
                    "resume_from_global_step": resume_from_global_step,
                }
            )

            # Metrics table(s).
            for rec in read_map["metrics"].records:
                step = _safe_int(rec.get("step"))
                data = rec.get("data")
                if step is None or not isinstance(data, dict):
                    continue
                raw_iteration = _safe_int(data.get("training/global_step"))
                canonical_step, _ = register_step_observation(
                    raw_step=step,
                    raw_iteration=raw_iteration,
                    source_table="step_metrics",
                )

                validation_logged = data.get("logging/validation_logged")
                if not isinstance(validation_logged, bool):
                    validation_logged = None
                if validation_logged is True and canonical_step is not None:
                    validation_steps_by_run[run_id].add(int(canonical_step))

                wide_key = (run_id, canonical_step if canonical_step is not None else step)
                if wide_key not in wide_rows_map:
                    wide_rows_map[wide_key] = {
                        "run_id": run_id,
                        "logical_run_group": logical_run_group,
                        "global_step": canonical_step if canonical_step is not None else step,
                        "global_step_canonical": canonical_step if canonical_step is not None else step,
                        "global_step_raw_stepfield": step,
                        "global_step_raw_iterationfield": raw_iteration,
                        "validation_logged": validation_logged,
                    }
                elif validation_logged is not None:
                    wide_rows_map[wide_key]["validation_logged"] = validation_logged

                for metric_key, metric_val in data.items():
                    metric_type, metric_value_float, metric_value_bool, metric_value_str = _classify_metric(metric_val)
                    step_metrics_long_rows.append(
                        {
                            "run_id": run_id,
                            "logical_run_group": logical_run_group,
                            "global_step": canonical_step if canonical_step is not None else step,
                            "global_step_canonical": canonical_step if canonical_step is not None else step,
                            "global_step_raw_stepfield": step,
                            "global_step_raw_iterationfield": raw_iteration,
                            "metric_key": metric_key,
                            "metric_type": metric_type,
                            "metric_value_float": metric_value_float,
                            "metric_value_bool": metric_value_bool,
                            "metric_value_str": metric_value_str,
                            "validation_logged": validation_logged,
                        }
                    )

                    if metric_key in curated_metric_cols:
                        col = curated_metric_cols[metric_key]
                        if isinstance(metric_val, bool):
                            wide_rows_map[wide_key][col] = metric_val
                        elif isinstance(metric_val, (int, float)):
                            wide_rows_map[wide_key][col] = float(metric_val)
                        elif metric_val is None:
                            wide_rows_map[wide_key][col] = None
                        else:
                            wide_rows_map[wide_key][col] = json.dumps(metric_val, ensure_ascii=True, default=_json_default)

            # Phase timings table.
            for rec in read_map["phase_timings"].records:
                raw_iteration = _safe_int(rec.get("iteration"))
                canonical_step, _ = register_step_observation(
                    raw_step=None,
                    raw_iteration=raw_iteration,
                    source_table="phase_timings",
                )
                row = dict(rec)
                row["run_id"] = run_id
                row["logical_run_group"] = logical_run_group
                row["global_step"] = canonical_step
                row["global_step_canonical"] = canonical_step
                row["global_step_raw_stepfield"] = None
                row["global_step_raw_iterationfield"] = raw_iteration
                phase_timings_rows.append(row)

            # Tokens and steps table.
            for rec in read_map["tokens"].records:
                raw_iteration = _safe_int(rec.get("iteration"))
                canonical_step, _ = register_step_observation(
                    raw_step=None,
                    raw_iteration=raw_iteration,
                    source_table="tokens_and_steps",
                )
                row = dict(rec)
                row["run_id"] = run_id
                row["logical_run_group"] = logical_run_group
                row["global_step"] = canonical_step
                row["global_step_canonical"] = canonical_step
                row["global_step_raw_stepfield"] = None
                row["global_step_raw_iterationfield"] = raw_iteration
                tokens_rows.append(row)

            # Hardware tables.
            def add_hw_rows(records: List[Dict[str, Any]], source: str, record_kind: str) -> None:
                target = hw_boundary_rows if record_kind == "boundary" else hw_periodic_rows
                for rec in records:
                    raw_iteration = _safe_int(rec.get("iteration"))
                    canonical_step, _ = register_step_observation(
                        raw_step=None,
                        raw_iteration=raw_iteration,
                        source_table=f"hardware_{record_kind}_{source}",
                    )
                    phase_name = rec.get("phase_name")

                    # Drop warmup idle rows by request.
                    if record_kind == "periodic" and canonical_step == 0 and phase_name == "idle":
                        continue

                    row = dict(rec)
                    row["run_id"] = run_id
                    row["logical_run_group"] = logical_run_group
                    row["global_step"] = canonical_step
                    row["global_step_canonical"] = canonical_step
                    row["global_step_raw_stepfield"] = None
                    row["global_step_raw_iterationfield"] = raw_iteration
                    row["source"] = source
                    row["record_kind"] = record_kind

                    if source == "nvml":
                        row["device_kind"] = "gpu"
                        row["device_id"] = row.get("gpu_uuid") or (f"gpu_index:{row.get('gpu_index')}" if row.get("gpu_index") is not None else None)
                        if active_nvml_device_keys is not None and row["device_id"] not in active_nvml_device_keys:
                            continue
                    else:
                        row["device_kind"] = "rapl"
                        row["device_id"] = row.get("rapl_domain") or row.get("domain_path")

                    row["phase_instance_id"] = _stable_hash(
                        run_id,
                        canonical_step,
                        row.get("phase_id"),
                        row.get("phase_name"),
                    )
                    row["boundary_pair_key"] = (
                        _stable_hash(row["phase_instance_id"], source, row.get("device_id"))
                        if record_kind == "boundary"
                        else None
                    )

                    # Convenience converted column while keeping raw units.
                    row["phase_domain_energy_delta_j"] = None
                    if "phase_domain_energy_delta_uJ" in row:
                        uj = _safe_float(row.get("phase_domain_energy_delta_uJ"))
                        row["phase_domain_energy_delta_j"] = (uj / 1_000_000.0) if uj is not None else None

                    target.append(row)
                    if record_kind == "boundary":
                        run_boundary_rows_local.append(row)

            add_hw_rows(read_map["nvml_boundary"].records, "nvml", "boundary")
            add_hw_rows(read_map["rapl_boundary"].records, "rapl", "boundary")
            add_hw_rows(read_map["nvml_periodic"].records, "nvml", "periodic")
            add_hw_rows(read_map["rapl_periodic"].records, "rapl", "periodic")

            (
                phase_rows_local,
                pair_rows_local,
                boundary_pair_integrity,
                boundary_pair_total,
                boundary_pair_valid,
                terminal_start_only_boundary_exception,
                terminal_start_only_invalid_pair_count,
            ) = build_boundary_integrity_outputs(run_id=run_id, run_boundary_rows_local=run_boundary_rows_local)
            phase_instance_rows.extend(phase_rows_local)
            boundary_pair_rows.extend(pair_rows_local)

            step_index_local_rows, run_checks = build_step_index_and_checks(
                run_id=run_id, run_step_observations=run_step_observations
            )
            step_index_rows.extend(step_index_local_rows)
            run_checks["boundary_pair_integrity"] = boundary_pair_integrity
            run_checks["boundary_pair_total"] = boundary_pair_total
            run_checks["boundary_pair_valid"] = boundary_pair_valid
            run_checks["terminal_start_only_boundary_exception"] = terminal_start_only_boundary_exception
            run_checks["terminal_start_only_invalid_pair_count"] = terminal_start_only_invalid_pair_count
            ingestion_checks_rows.append(run_checks)

            observed_steps = [
                _safe_int(obs.get("canonical_step"))
                for obs in run_step_observations
                if _safe_int(obs.get("canonical_step")) is not None
            ]
            observed_total_steps = max(observed_steps) if observed_steps else None
            run_row["observed_total_steps"] = observed_total_steps
            if run_row["total_steps"] is None and observed_total_steps is not None:
                run_row["total_steps"] = observed_total_steps
                run_row["total_steps_from_observed_fallback"] = True
            runs_rows.append(run_row)

            report_rows.append(
                {
                    **quality.__dict__,
                    "included": True,
                }
            )

        except Exception as exc:
            quality.reason = f"unexpected_error:{type(exc).__name__}"
            report_rows.append(
                {
                    **quality.__dict__,
                    "included": False,
                }
            )

    # Build DataFrames.
    runs_df = pd.DataFrame(runs_rows)
    lineage_df = pd.DataFrame(lineage_rows)
    step_index_map_df = pd.DataFrame(step_index_rows)
    ingestion_checks_df = pd.DataFrame(ingestion_checks_rows)
    phase_instances_df = pd.DataFrame(phase_instance_rows)
    boundary_pair_integrity_df = pd.DataFrame(boundary_pair_rows)
    metrics_long_df = pd.DataFrame(step_metrics_long_rows)
    phase_timings_df = pd.DataFrame(phase_timings_rows)
    tokens_df = pd.DataFrame(tokens_rows)
    hw_boundary_df = pd.DataFrame(hw_boundary_rows)
    hw_periodic_df = pd.DataFrame(hw_periodic_rows)
    report_df = pd.DataFrame(report_rows)
    phase_fact_df = pd.DataFrame()

    wide_rows = list(wide_rows_map.values())
    wide_df = pd.DataFrame(wide_rows)

    # Derive phase summary from hardware tables.
    phase_group_cols = ["run_id", "global_step_canonical", "phase_name", "phase_id", "source"]

    boundary_summary = pd.DataFrame(columns=phase_group_cols)
    if not hw_boundary_df.empty:
        b = hw_boundary_df.copy()
        for col in ["phase_duration_s", "phase_gpu_energy_delta_J", "phase_domain_energy_delta_uJ", "phase_domain_energy_delta_j"]:
            if col in b.columns:
                b[col] = pd.to_numeric(b[col], errors="coerce")

        agg_spec: Dict[str, Any] = {"device_id": "nunique"}
        rename_map = {"device_id": "boundary_device_count"}
        if "phase_duration_s" in b.columns:
            agg_spec["phase_duration_s"] = "max"
            rename_map["phase_duration_s"] = "boundary_phase_duration_s_max"
        if "phase_gpu_energy_delta_J" in b.columns:
            agg_spec["phase_gpu_energy_delta_J"] = "sum"
            rename_map["phase_gpu_energy_delta_J"] = "boundary_gpu_energy_delta_j_sum"
        if "phase_domain_energy_delta_uJ" in b.columns:
            agg_spec["phase_domain_energy_delta_uJ"] = "sum"
            rename_map["phase_domain_energy_delta_uJ"] = "boundary_rapl_energy_delta_uj_sum"
        if "phase_domain_energy_delta_j" in b.columns:
            agg_spec["phase_domain_energy_delta_j"] = "sum"
            rename_map["phase_domain_energy_delta_j"] = "boundary_rapl_energy_delta_j_sum"

        boundary_summary = b.groupby(phase_group_cols, dropna=False).agg(agg_spec).reset_index().rename(columns=rename_map)
        boundary_counts = b.groupby(phase_group_cols, dropna=False).size().reset_index(name="boundary_row_count")
        boundary_summary = boundary_summary.merge(boundary_counts, on=phase_group_cols, how="outer")

    periodic_summary = pd.DataFrame(columns=phase_group_cols)
    if not hw_periodic_df.empty:
        p = hw_periodic_df.copy()
        p = filter_periodic_to_phase_window(p, phase_instances_df)
        p = add_time_weights(
            p,
            group_cols=["phase_instance_id", "source", "device_id"],
            ts_col="ts_monotonic_ns",
            out_col="sample_weight_ns",
        )

        numeric_metric_map = {
            "periodic_gpu_power_mw": "gpu_power_mW",
            "periodic_gpu_util_pct": "gpu_util_pct",
            "periodic_sm_util_pct": "sm_util_pct",
            "periodic_mem_util_pct": "mem_util_pct",
            "periodic_temp_gpu_c": "temp_gpu_C",
            "periodic_cpu_energy_uj": "cpu_energy_uJ",
        }
        bool_metric_map = {
            "periodic_thr_sw_power_cap_frac": "thr_sw_power_cap",
            "periodic_thr_thermal_slowdown_frac": "thr_thermal_slowdown",
            "periodic_thr_hw_slowdown_frac": "thr_hw_slowdown",
            "periodic_thr_hw_power_brake_frac": "thr_hw_power_brake",
        }
        periodic_summary = aggregate_periodic_metrics(
            p,
            group_cols=phase_group_cols,
            numeric_metric_map=numeric_metric_map,
            bool_metric_map=bool_metric_map,
            weight_col="sample_weight_ns",
            include_sample_mean=True,
            include_time_weighted_mean=True,
            sample_count_col="periodic_row_count",
        )
        periodic_device_counts = (
            p.groupby(phase_group_cols, dropna=False)["device_id"].nunique().reset_index(name="periodic_device_count")
        )
        periodic_summary = periodic_summary.merge(periodic_device_counts, on=phase_group_cols, how="left")

    if not boundary_summary.empty and not periodic_summary.empty:
        phase_summary_df = boundary_summary.merge(periodic_summary, on=phase_group_cols, how="outer")
    elif not boundary_summary.empty:
        phase_summary_df = boundary_summary
    elif not periodic_summary.empty:
        phase_summary_df = periodic_summary
    else:
        phase_summary_df = pd.DataFrame(columns=phase_group_cols)

    # Ensure key columns exist even for empty outputs.
    runs_df = _ensure_columns(runs_df, ["run_id", "run_dir", "logical_run_group", "is_resumed_run"])
    lineage_df = _ensure_columns(lineage_df, ["run_id", "is_resumed_run", "resume_parent_run_name", "resume_from_global_step"])
    step_index_map_df = _ensure_columns(
        step_index_map_df,
        ["run_id", "raw_step", "raw_iteration", "canonical_step", "mismatch_flag", "observation_count"],
    )
    ingestion_checks_df = _ensure_columns(
        ingestion_checks_df,
        [
            "run_id",
            "raw_step_distinct_count",
            "raw_iteration_distinct_count",
            "overlap_distinct_count",
            "raw_step_only_count",
            "raw_iteration_only_count",
            "mismatch_count",
            "observation_count",
            "mismatch_rate",
            "join_coverage_rate",
            "boundary_pair_integrity",
            "boundary_pair_total",
            "boundary_pair_valid",
            "terminal_start_only_boundary_exception",
            "terminal_start_only_invalid_pair_count",
        ],
    )
    phase_instances_df = _ensure_columns(
        phase_instances_df,
        [
            "run_id",
            "global_step",
            "global_step_canonical",
            "global_step_raw_stepfield",
            "global_step_raw_iterationfield",
            "phase_name",
            "phase_id",
            "phase_instance_id",
            "phase_start_ts_monotonic_ns",
            "phase_end_ts_monotonic_ns",
            "start_row_count",
            "end_row_count",
            "boundary_row_count",
        ],
    )
    boundary_pair_integrity_df = _ensure_columns(
        boundary_pair_integrity_df,
        [
            "run_id",
            "boundary_pair_key",
            "phase_instance_id",
            "source",
            "device_id",
            "global_step_canonical",
            "start_count",
            "end_count",
            "row_count",
            "is_valid_pair",
            "is_terminal_start_only_pair",
        ],
    )
    metrics_long_df = _ensure_columns(
        metrics_long_df,
        [
            "run_id",
            "logical_run_group",
            "global_step",
            "global_step_canonical",
            "global_step_raw_stepfield",
            "global_step_raw_iterationfield",
            "metric_key",
            "metric_type",
            "metric_value_float",
            "metric_value_bool",
            "metric_value_str",
            "validation_logged",
        ],
    )
    wide_df = _ensure_columns(
        wide_df,
        [
            "run_id",
            "logical_run_group",
            "global_step",
            "global_step_canonical",
            "global_step_raw_stepfield",
            "global_step_raw_iterationfield",
            "validation_logged",
            "is_warmup_idle",
            "is_validation_step",
            "is_incomplete_phase",
            "is_outlier_sample",
        ],
    )
    for key in CURATED_METRIC_KEYS:
        wide_df = _ensure_columns(wide_df, [curated_metric_cols[key]])

    phase_timings_df = _ensure_columns(
        phase_timings_df,
        [
            "run_id",
            "logical_run_group",
            "global_step",
            "global_step_canonical",
            "global_step_raw_stepfield",
            "global_step_raw_iterationfield",
            "phase_name",
            "phase_id",
            "subphase_name",
            "value",
            "metric_unit",
        ],
    )
    tokens_df = _ensure_columns(
        tokens_df,
        [
            "run_id",
            "logical_run_group",
            "global_step",
            "global_step_canonical",
            "global_step_raw_stepfield",
            "global_step_raw_iterationfield",
            "phase_name",
            "phase_id",
            "metric_scope",
        ],
    )
    hw_boundary_df = _ensure_columns(
        hw_boundary_df,
        [
            "run_id",
            "logical_run_group",
            "global_step",
            "global_step_canonical",
            "global_step_raw_stepfield",
            "global_step_raw_iterationfield",
            "phase_name",
            "phase_id",
            "phase_instance_id",
            "boundary_pair_key",
            "source",
            "record_kind",
            "device_kind",
            "device_id",
            "phase_event",
            "ts_monotonic_ns",
            "ts_wall_ms",
        ],
    )
    hw_periodic_df = _ensure_columns(
        hw_periodic_df,
        [
            "run_id",
            "logical_run_group",
            "global_step",
            "global_step_canonical",
            "global_step_raw_stepfield",
            "global_step_raw_iterationfield",
            "phase_name",
            "phase_id",
            "phase_instance_id",
            "boundary_pair_key",
            "source",
            "record_kind",
            "device_kind",
            "device_id",
            "ts_monotonic_ns",
            "ts_wall_ms",
        ],
    )
    phase_summary_df = _ensure_columns(
        phase_summary_df,
        phase_group_cols
        + [
            "global_step",
            "global_step_raw_stepfield",
            "global_step_raw_iterationfield",
            "phase_instance_id",
            "is_warmup_idle",
            "is_validation_step",
            "is_incomplete_phase",
            "is_outlier_sample",
        ],
    )
    phase_summary_periodic_cols = [
        "periodic_gpu_power_mw_sample_mean",
        "periodic_gpu_power_mw_twa",
        "periodic_gpu_util_pct_sample_mean",
        "periodic_gpu_util_pct_twa",
        "periodic_sm_util_pct_sample_mean",
        "periodic_sm_util_pct_twa",
        "periodic_mem_util_pct_sample_mean",
        "periodic_mem_util_pct_twa",
        "periodic_temp_gpu_c_sample_mean",
        "periodic_temp_gpu_c_twa",
        "periodic_cpu_energy_uj_sample_mean",
        "periodic_cpu_energy_uj_twa",
        "periodic_thr_sw_power_cap_frac_sample_mean",
        "periodic_thr_sw_power_cap_frac_twa",
        "periodic_thr_thermal_slowdown_frac_sample_mean",
        "periodic_thr_thermal_slowdown_frac_twa",
        "periodic_thr_hw_slowdown_frac_sample_mean",
        "periodic_thr_hw_slowdown_frac_twa",
        "periodic_thr_hw_power_brake_frac_sample_mean",
        "periodic_thr_hw_power_brake_frac_twa",
        "periodic_row_count",
        "periodic_device_count",
    ]
    phase_summary_df = _ensure_columns(phase_summary_df, phase_summary_periodic_cols)
    report_df = _ensure_columns(
        report_df,
        [
            "run_dir",
            "run_id",
            "status",
            "reason",
            "included",
            "missing_files",
            "zero_line_files",
            "parse_error_files",
            "analysis_mask_is_warmup_idle_count",
            "analysis_mask_is_validation_step_count",
            "analysis_mask_is_incomplete_phase_count",
            "analysis_mask_is_outlier_sample_count",
        ],
    )

    # Keep both canonical and raw step fields visible in phase summary.
    phase_summary_df["global_step"] = phase_summary_df["global_step_canonical"]
    phase_summary_df["global_step_raw_stepfield"] = None
    phase_summary_df["global_step_raw_iterationfield"] = phase_summary_df["global_step_canonical"]

    phase_outlier_flags_df = build_phase_outlier_flags(hw_boundary_df)

    # Canonical one-row-per-phase-instance fact table.
    phase_fact_df = _build_phase_fact(
        runs_df=runs_df,
        phase_instances_df=phase_instances_df,
        hw_boundary_df=hw_boundary_df,
        hw_periodic_df=hw_periodic_df,
        tokens_df=tokens_df,
        validation_steps_by_run=validation_steps_by_run,
        phase_outlier_flags_df=phase_outlier_flags_df,
    )

    # Propagate analysis masks to other phase/step outputs as appropriate.
    phase_mask_cols = [
        "run_id",
        "phase_instance_id",
        "is_warmup_idle",
        "is_validation_step",
        "is_incomplete_phase",
        "is_outlier_sample",
    ]
    phase_mask_df = phase_fact_df[phase_mask_cols].copy() if not phase_fact_df.empty else pd.DataFrame(columns=phase_mask_cols)

    if not phase_summary_df.empty:
        phase_summary_df["phase_instance_id"] = phase_summary_df.apply(
            lambda r: _stable_hash(r.get("run_id"), r.get("global_step_canonical"), r.get("phase_id"), r.get("phase_name")),
            axis=1,
        )
        phase_summary_df = phase_summary_df.merge(phase_mask_df, on=["run_id", "phase_instance_id"], how="left")
        for c in ["is_warmup_idle", "is_validation_step", "is_incomplete_phase", "is_outlier_sample"]:
            phase_summary_df = _coalesce_join_column(phase_summary_df, c, default=False, cast="bool")

    if not wide_df.empty:
        val_map = (
            phase_fact_df.groupby(["run_id", "global_step_canonical"], dropna=False)["is_validation_step"]
            .max()
            .reset_index()
            if not phase_fact_df.empty
            else pd.DataFrame(columns=["run_id", "global_step_canonical", "is_validation_step"])
        )
        wide_df = wide_df.merge(val_map, on=["run_id", "global_step_canonical"], how="left")
        if "is_validation_step" not in wide_df.columns:
            if "is_validation_step_y" in wide_df.columns:
                wide_df["is_validation_step"] = wide_df["is_validation_step_y"]
            elif "is_validation_step_x" in wide_df.columns:
                wide_df["is_validation_step"] = wide_df["is_validation_step_x"]
            else:
                wide_df["is_validation_step"] = False
        wide_df["is_validation_step"] = wide_df["is_validation_step"].fillna(False).astype(bool)
        wide_df["is_warmup_idle"] = (
            (pd.to_numeric(wide_df["global_step_canonical"], errors="coerce") == 0)
            & (wide_df.get("phase_name", pd.Series([None] * len(wide_df))) == "idle")
        )
        step_incomplete = (
            phase_fact_df.groupby(["run_id", "global_step_canonical"], dropna=False)["is_incomplete_phase"]
            .max()
            .reset_index()
            if not phase_fact_df.empty
            else pd.DataFrame(columns=["run_id", "global_step_canonical", "is_incomplete_phase"])
        )
        step_outlier = (
            phase_fact_df.groupby(["run_id", "global_step_canonical"], dropna=False)["is_outlier_sample"]
            .max()
            .reset_index()
            if not phase_fact_df.empty
            else pd.DataFrame(columns=["run_id", "global_step_canonical", "is_outlier_sample"])
        )
        wide_df = wide_df.merge(step_incomplete, on=["run_id", "global_step_canonical"], how="left")
        wide_df = wide_df.merge(step_outlier, on=["run_id", "global_step_canonical"], how="left")
        for c in ["is_incomplete_phase", "is_outlier_sample"]:
            if c not in wide_df.columns:
                if f"{c}_y" in wide_df.columns:
                    wide_df[c] = wide_df[f"{c}_y"]
                elif f"{c}_x" in wide_df.columns:
                    wide_df[c] = wide_df[f"{c}_x"]
                else:
                    wide_df[c] = False
            wide_df[c] = wide_df[c].fillna(False).astype(bool)
        for c in ["is_warmup_idle", "is_validation_step", "is_incomplete_phase", "is_outlier_sample"]:
            wide_df = _coalesce_join_column(wide_df, c, default=False, cast="bool")

    if not report_df.empty and not phase_fact_df.empty:
        run_mask_counts = (
            phase_fact_df.groupby("run_id", dropna=False)
            .agg(
                analysis_mask_is_warmup_idle_count=("is_warmup_idle", "sum"),
                analysis_mask_is_validation_step_count=("is_validation_step", "sum"),
                analysis_mask_is_incomplete_phase_count=("is_incomplete_phase", "sum"),
                analysis_mask_is_outlier_sample_count=("is_outlier_sample", "sum"),
            )
            .reset_index()
        )
        report_df = report_df.merge(run_mask_counts, on="run_id", how="left")
        for c in [
            "analysis_mask_is_warmup_idle_count",
            "analysis_mask_is_validation_step_count",
            "analysis_mask_is_incomplete_phase_count",
            "analysis_mask_is_outlier_sample_count",
        ]:
            if c not in report_df.columns:
                left = f"{c}_x"
                right = f"{c}_y"
                if left in report_df.columns and right in report_df.columns:
                    report_df[c] = pd.to_numeric(report_df[right], errors="coerce").fillna(
                        pd.to_numeric(report_df[left], errors="coerce")
                    )
                elif right in report_df.columns:
                    report_df[c] = report_df[right]
                elif left in report_df.columns:
                    report_df[c] = report_df[left]
                else:
                    report_df[c] = 0
    if not report_df.empty:
        for c in [
            "analysis_mask_is_warmup_idle_count",
            "analysis_mask_is_validation_step_count",
            "analysis_mask_is_incomplete_phase_count",
            "analysis_mask_is_outlier_sample_count",
        ]:
            report_df = _coalesce_join_column(report_df, c, default=0, cast="int")

    # Make absence explicit for count-like phase summary fields.
    if not phase_summary_df.empty:
        for c in [
            "boundary_device_count",
            "boundary_row_count",
            "periodic_row_count",
            "periodic_device_count",
        ]:
            if c in phase_summary_df.columns:
                phase_summary_df[c] = pd.to_numeric(phase_summary_df[c], errors="coerce").fillna(0).astype(int)

    # Hard-fail invariants are centralized in ingestion_checks.py.
    total_mismatch_count, min_boundary_pair_integrity = validate_ingestion_checks(ingestion_checks_df)

    # Stable sort for reproducibility.
    if not runs_df.empty:
        runs_df = runs_df.sort_values(["run_id"]).reset_index(drop=True)
    if not lineage_df.empty:
        lineage_df = lineage_df.sort_values(["run_id"]).reset_index(drop=True)
    if not step_index_map_df.empty:
        step_index_map_df = step_index_map_df.sort_values(
            ["run_id", "canonical_step", "raw_step", "raw_iteration", "mismatch_flag"]
        ).reset_index(drop=True)
    if not ingestion_checks_df.empty:
        ingestion_checks_df = ingestion_checks_df.sort_values(["run_id"]).reset_index(drop=True)
    if not phase_instances_df.empty:
        phase_instances_df = phase_instances_df.sort_values(
            ["run_id", "global_step_canonical", "phase_id", "phase_name"]
        ).reset_index(drop=True)
    if not boundary_pair_integrity_df.empty:
        boundary_pair_integrity_df = boundary_pair_integrity_df.sort_values(
            ["run_id", "phase_instance_id", "source", "device_id"]
        ).reset_index(drop=True)
    if not metrics_long_df.empty:
        metrics_long_df = metrics_long_df.sort_values(["run_id", "global_step", "metric_key"]).reset_index(drop=True)
    if not wide_df.empty:
        wide_df = wide_df.sort_values(["run_id", "global_step"]).reset_index(drop=True)
    if not phase_timings_df.empty:
        phase_timings_df = phase_timings_df.sort_values(
            ["run_id", "global_step", "phase_name", "subphase_name"]
        ).reset_index(drop=True)
    if not tokens_df.empty:
        tokens_df = tokens_df.sort_values(["run_id", "global_step", "phase_name"]).reset_index(drop=True)
    if not hw_boundary_df.empty:
        hw_boundary_df = hw_boundary_df.sort_values(
            ["run_id", "global_step", "phase_name", "phase_event", "source", "device_id", "ts_monotonic_ns"]
        ).reset_index(drop=True)
    if not hw_periodic_df.empty:
        hw_periodic_df = hw_periodic_df.sort_values(
            ["run_id", "global_step", "phase_name", "source", "device_id", "ts_monotonic_ns"]
        ).reset_index(drop=True)
    if not phase_summary_df.empty:
        phase_summary_df = phase_summary_df.sort_values(phase_group_cols).reset_index(drop=True)
    if not phase_fact_df.empty:
        phase_fact_df = phase_fact_df.sort_values(
            ["run_id", "global_step_canonical", "phase_id", "phase_name", "phase_instance_id"]
        ).reset_index(drop=True)
    if not report_df.empty:
        for c in [
            "analysis_mask_is_warmup_idle_count",
            "analysis_mask_is_validation_step_count",
            "analysis_mask_is_incomplete_phase_count",
            "analysis_mask_is_outlier_sample_count",
        ]:
            report_df[c] = pd.to_numeric(report_df[c], errors="coerce").fillna(0).astype(int)
        report_df = report_df.sort_values(["included", "run_id", "run_dir"], ascending=[False, True, True]).reset_index(drop=True)

    analysis_views = _build_analysis_views(
        runs_df=runs_df,
        lineage_df=lineage_df,
        phase_fact_df=phase_fact_df,
        wide_df=wide_df,
        hw_periodic_df=hw_periodic_df,
        hw_boundary_df=hw_boundary_df,
        ingestion_checks_df=ingestion_checks_df,
        report_df=report_df,
    )
    phase_fact_view_df = analysis_views["phase_fact_view"]
    step_fact_view_df = analysis_views["step_fact_view"]
    run_summary_view_df = analysis_views["run_summary_view"]
    comparison_view_df = analysis_views["comparison_view"]
    device_timeseries_view_df = analysis_views["device_timeseries_view"]
    integrity_view_df = analysis_views["integrity_view"]

    if not phase_fact_view_df.empty:
        phase_fact_view_df = phase_fact_view_df.sort_values(
            ["run_id", "global_step_canonical", "phase_id", "phase_name", "phase_instance_id"]
        ).reset_index(drop=True)
    if not step_fact_view_df.empty:
        step_fact_view_df = step_fact_view_df.sort_values(
            ["run_id", "global_step_canonical"]
        ).reset_index(drop=True)
    if not run_summary_view_df.empty:
        run_summary_view_df = run_summary_view_df.sort_values(["run_id"]).reset_index(drop=True)
    if not comparison_view_df.empty:
        comparison_view_df = comparison_view_df.sort_values(
            ["policy", "model", "experiment_variant", "is_checkpoint_continuation"]
        ).reset_index(drop=True)
    if not device_timeseries_view_df.empty:
        device_timeseries_view_df = device_timeseries_view_df.sort_values(
            ["run_id", "global_step_canonical", "phase_name", "phase_id", "device_kind", "device_id", "ts_monotonic_ns"]
        ).reset_index(drop=True)
    if not integrity_view_df.empty:
        integrity_view_df = integrity_view_df.sort_values(["run_id"]).reset_index(drop=True)

    # Serialize list/dict report fields as JSON strings for Parquet compatibility.
    for col in ["missing_files", "zero_line_files", "parse_error_files"]:
        report_df[col] = report_df[col].apply(lambda x: json.dumps(x, ensure_ascii=True, default=_json_default))

    outputs = {
        "runs": runs_df,
        "run_lineage": lineage_df,
        "step_index_map": step_index_map_df,
        "ingestion_checks": ingestion_checks_df,
        "phase_instances": phase_instances_df,
        "boundary_pair_integrity": boundary_pair_integrity_df,
        "step_metrics_long": metrics_long_df,
        "step_metrics_wide_curated": wide_df,
        "phase_timings_long": phase_timings_df,
        "tokens_and_steps": tokens_df,
        "hardware_boundary": hw_boundary_df,
        "hardware_periodic": hw_periodic_df,
        "phase_summary": phase_summary_df,
        "phase_fact": phase_fact_df,
        "ingestion_report": report_df,
        "phase_fact_view": phase_fact_view_df,
        "step_fact_view": step_fact_view_df,
        "run_summary_view": run_summary_view_df,
        "comparison_view": comparison_view_df,
        "device_timeseries_view": device_timeseries_view_df,
        "integrity_view": integrity_view_df,
    }

    for key, df in outputs.items():
        out_path = output_root / TABLE_FILES[key]
        df.to_parquet(out_path, index=False)

    included_runs = int(report_df["included"].astype(bool).sum()) if not report_df.empty else 0
    excluded_runs = int((~report_df["included"].astype(bool)).sum()) if not report_df.empty else 0
    min_join_coverage = (
        float(pd.to_numeric(ingestion_checks_df["join_coverage_rate"], errors="coerce").min())
        if not ingestion_checks_df.empty
        else 1.0
    )

    print("Dataset build complete")
    print(f"  Results root: {results_root}")
    if include_dirs:
        print(f"  Included subdirs: {len(include_dirs)}")
        for include_dir in include_dirs:
            print(f"    - {include_dir}")
    print(f"  Output root:  {output_root}")
    print(f"  Included runs: {included_runs}")
    print(f"  Excluded runs: {excluded_runs}")
    print(f"  Total mismatch count: {total_mismatch_count}")
    print(f"  Min join coverage rate: {min_join_coverage:.6f}")
    print(f"  Min boundary pair integrity: {min_boundary_pair_integrity:.6f}")
    print("  Wrote tables:")
    for key in TABLE_FILES:
        print(f"    - {TABLE_FILES[key]}")


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not results_root.exists() or not results_root.is_dir():
        raise FileNotFoundError(f"results root does not exist or is not a directory: {results_root}")

    # Currently single-process even if workers > 1; keep interface stable.
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    include_dirs = _resolve_include_subdirs(results_root=results_root, include_subdirs=args.include_subdir)

    build_datasets(
        results_root=results_root,
        output_root=output_root,
        overwrite=args.overwrite,
        include_dirs=include_dirs,
    )


if __name__ == "__main__":
    main()
