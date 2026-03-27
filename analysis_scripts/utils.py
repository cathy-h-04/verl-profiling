"""Utility helpers for dataset ingestion and transformation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from ingestion_checks import validate_phase_fact
from periodic_aggregations import (
    add_time_weights,
    aggregate_periodic_metrics,
    filter_periodic_to_phase_window,
)


PHASE_FACT_STRICT_COLUMNS = [
    "run_id",
    "logical_run_group",
    "global_step",
    "global_step_canonical",
    "global_step_raw_stepfield",
    "global_step_raw_iterationfield",
    "phase_id",
    "phase_name",
    "phase_instance_id",
    "phase_start_ts_monotonic_ns",
    "phase_end_ts_monotonic_ns",
    "phase_duration_s_canonical",
    "gpu_energy_j",
    "cpu_energy_j",
    "dram_energy_j",
    "total_energy_j",
    "gpu_device_count",
    "rapl_device_count",
    "token_rollout_total_tokens",
    "token_rollout_output_tokens_total",
    "token_rollout_prompt_tokens_total",
    "token_train_batch_tokens",
    "token_train_tokens_effective_estimated",
    "token_train_microbatch_tokens_estimated",
    "token_row_present",
    "shape_gpu_power_mw_twa",
    "shape_gpu_power_mw_sample_mean",
    "shape_gpu_util_pct_twa",
    "shape_gpu_util_pct_sample_mean",
    "shape_sm_util_pct_twa",
    "shape_sm_util_pct_sample_mean",
    "shape_mem_util_pct_twa",
    "shape_mem_util_pct_sample_mean",
    "shape_sm_clock_mhz_twa",
    "shape_sm_clock_mhz_sample_mean",
    "shape_mem_clock_mhz_twa",
    "shape_mem_clock_mhz_sample_mean",
    "shape_temp_gpu_c_twa",
    "shape_temp_gpu_c_sample_mean",
    "shape_pcie_tx_bytes_s_twa",
    "shape_pcie_tx_bytes_s_sample_mean",
    "shape_pcie_rx_bytes_s_twa",
    "shape_pcie_rx_bytes_s_sample_mean",
    "shape_thr_sw_power_cap_frac_twa",
    "shape_thr_sw_power_cap_frac_sample_mean",
    "shape_thr_thermal_slowdown_frac_twa",
    "shape_thr_thermal_slowdown_frac_sample_mean",
    "shape_thr_hw_slowdown_frac_twa",
    "shape_thr_hw_slowdown_frac_sample_mean",
    "shape_thr_hw_power_brake_frac_twa",
    "shape_thr_hw_power_brake_frac_sample_mean",
    "shape_periodic_sample_count",
    "is_warmup_idle",
    "is_validation_step",
    "is_incomplete_phase",
    "is_outlier_sample",
]


@dataclass
class JsonlReadResult:
    records: List[Dict[str, Any]]
    parse_errors: int
    nonempty_lines: int


def _sanitize_col(metric_key: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", metric_key).strip("_").lower()
    return f"metric_{name}" if name else "metric_unknown"


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {"_value": obj}


def _read_jsonl(path: Path) -> JsonlReadResult:
    records: List[Dict[str, Any]] = []
    parse_errors = 0
    nonempty = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            nonempty += 1
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    records.append({"_value": obj})
            except Exception:
                parse_errors += 1

    return JsonlReadResult(records=records, parse_errors=parse_errors, nonempty_lines=nonempty)


def _parse_slurm_job_ids(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            out[key.strip()] = val.strip()
    return out


def _discover_run_dirs(results_root: Path, include_dirs: Optional[Iterable[Path]] = None) -> List[Path]:
    search_roots = [results_root] if not include_dirs else list(include_dirs)

    run_dirs = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        if (root / "experiment_name.txt").exists():
            run_dirs.add(root)
        run_dirs.update(p.parent for p in root.rglob("experiment_name.txt"))

    return sorted(p for p in run_dirs if p.is_dir())


def _required_files_for_run(run_id: str, required_static_files: Iterable[str]) -> List[str]:
    return list(required_static_files) + [
        f"{run_id}.jsonl",
        f"{run_id}_config.json",
        f"phase_timings_{run_id}.jsonl",
    ]


def _extract_lineage(resume_path: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    if not resume_path:
        return None, None
    m = re.search(r"/checkpoints/(.+?)/global_step_(\d+)$", resume_path)
    if not m:
        return None, None
    parent_run_name, step = m.group(1), m.group(2)
    return parent_run_name, _safe_int(step)


def _json_default(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    return str(x)


def _ensure_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> pd.DataFrame:
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    return df


def _classify_metric(value: Any) -> Tuple[str, Optional[float], Optional[bool], Optional[str]]:
    if value is None:
        return "null", None, None, None
    if isinstance(value, bool):
        return "bool", None, value, None
    if isinstance(value, (int, float)):
        return "number", float(value), None, None
    if isinstance(value, str):
        return "string", None, None, value
    try:
        encoded = json.dumps(value, ensure_ascii=True, default=_json_default)
    except Exception:
        encoded = str(value)
    return "other", None, None, encoded


def _stable_hash(*parts: Any) -> str:
    """Generate a stable hash for composite identities."""
    normalized = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _canonicalize_step(
    raw_step: Optional[int], raw_iteration: Optional[int]
) -> Tuple[Optional[int], bool]:
    """Return canonical step and mismatch flag from raw step fields."""
    if raw_step is not None and raw_iteration is not None:
        return raw_step, raw_step != raw_iteration
    if raw_step is not None:
        return raw_step, False
    if raw_iteration is not None:
        return raw_iteration, False
    return None, False


def _build_phase_fact(
    runs_df: pd.DataFrame,
    phase_instances_df: pd.DataFrame,
    hw_boundary_df: pd.DataFrame,
    hw_periodic_df: pd.DataFrame,
    tokens_df: pd.DataFrame,
    validation_steps_by_run: Dict[str, set[int]],
    phase_outlier_flags_df: pd.DataFrame,
) -> pd.DataFrame:
    if phase_instances_df.empty:
        return pd.DataFrame(columns=PHASE_FACT_STRICT_COLUMNS)

    phase_fact = phase_instances_df[
        [
            "run_id",
            "global_step",
            "global_step_canonical",
            "global_step_raw_stepfield",
            "global_step_raw_iterationfield",
            "phase_id",
            "phase_name",
            "phase_instance_id",
            "phase_start_ts_monotonic_ns",
            "phase_end_ts_monotonic_ns",
        ]
    ].copy()

    phase_fact = phase_fact.merge(
        runs_df[["run_id", "logical_run_group"]].drop_duplicates(),
        on="run_id",
        how="left",
    )

    start_ns = pd.to_numeric(phase_fact["phase_start_ts_monotonic_ns"], errors="coerce")
    end_ns = pd.to_numeric(phase_fact["phase_end_ts_monotonic_ns"], errors="coerce")
    duration = (end_ns - start_ns) / 1_000_000_000.0
    phase_fact["phase_duration_s_canonical"] = duration.where(duration >= 0, None)

    b_end = hw_boundary_df[hw_boundary_df["phase_event"] == "END"].copy()
    if not b_end.empty:
        b_end["phase_gpu_energy_delta_J"] = pd.to_numeric(b_end.get("phase_gpu_energy_delta_J"), errors="coerce")
        b_end["phase_domain_energy_delta_j"] = pd.to_numeric(b_end.get("phase_domain_energy_delta_j"), errors="coerce")

        nv = b_end[b_end["source"] == "nvml"]
        nv_agg = pd.DataFrame(columns=["run_id", "phase_instance_id"])
        if not nv.empty:
            nv_agg = (
                nv.groupby(["run_id", "phase_instance_id"], dropna=False)
                .agg(gpu_energy_j=("phase_gpu_energy_delta_J", "sum"), gpu_device_count=("device_id", "nunique"))
                .reset_index()
            )

        rp = b_end[b_end["source"] == "rapl"].copy()
        rp_agg = pd.DataFrame(columns=["run_id", "phase_instance_id"])
        if not rp.empty:
            rp["rapl_domain_lc"] = rp["rapl_domain"].astype(str).str.lower() if "rapl_domain" in rp.columns else ""
            rp["dram_j"] = rp["phase_domain_energy_delta_j"].where(rp["rapl_domain_lc"].str.startswith("dram"), 0.0)
            rp["cpu_j"] = rp["phase_domain_energy_delta_j"].where(~rp["rapl_domain_lc"].str.startswith("dram"), 0.0)
            rp_agg = (
                rp.groupby(["run_id", "phase_instance_id"], dropna=False)
                .agg(
                    cpu_energy_j=("cpu_j", "sum"),
                    dram_energy_j=("dram_j", "sum"),
                    rapl_device_count=("device_id", "nunique"),
                )
                .reset_index()
            )

        phase_fact = phase_fact.merge(nv_agg, on=["run_id", "phase_instance_id"], how="left")
        phase_fact = phase_fact.merge(rp_agg, on=["run_id", "phase_instance_id"], how="left")
    else:
        phase_fact["gpu_energy_j"] = None
        phase_fact["gpu_device_count"] = None
        phase_fact["cpu_energy_j"] = None
        phase_fact["dram_energy_j"] = None
        phase_fact["rapl_device_count"] = None

    ecols = ["gpu_energy_j", "cpu_energy_j", "dram_energy_j"]
    for c in ecols:
        if c not in phase_fact.columns:
            phase_fact[c] = None
        phase_fact[c] = pd.to_numeric(phase_fact[c], errors="coerce")
    has_any_energy = phase_fact[ecols].notna().any(axis=1)
    phase_fact["total_energy_j"] = phase_fact[ecols].fillna(0.0).sum(axis=1).where(has_any_energy, None)

    phase_fact["is_warmup_idle"] = (
        (pd.to_numeric(phase_fact["global_step_canonical"], errors="coerce") == 0)
        & (phase_fact["phase_name"] == "idle")
    )
    phase_fact["is_validation_step"] = False
    for run_id, steps in validation_steps_by_run.items():
        if not steps:
            continue
        mask = (phase_fact["run_id"] == run_id) & (
            pd.to_numeric(phase_fact["global_step_canonical"], errors="coerce").isin(sorted(steps))
        )
        phase_fact.loc[mask, "is_validation_step"] = True
    phase_fact.loc[phase_fact["phase_name"] == "validation", "is_validation_step"] = True

    phase_fact["is_incomplete_phase"] = (
        phase_fact["phase_start_ts_monotonic_ns"].isna()
        | phase_fact["phase_end_ts_monotonic_ns"].isna()
        | pd.to_numeric(phase_fact["phase_duration_s_canonical"], errors="coerce").isna()
        | (pd.to_numeric(phase_fact["phase_duration_s_canonical"], errors="coerce") <= 0)
    )

    token_map = {
        "rollout_total_tokens": "token_rollout_total_tokens",
        "rollout_output_tokens_total": "token_rollout_output_tokens_total",
        "rollout_prompt_tokens_total": "token_rollout_prompt_tokens_total",
        "train_batch_tokens": "token_train_batch_tokens",
        "train_tokens_effective_estimated": "token_train_tokens_effective_estimated",
        "train_microbatch_tokens_estimated": "token_train_microbatch_tokens_estimated",
    }
    tok_cols = [c for c in token_map if c in tokens_df.columns]
    if tok_cols:
        tok = tokens_df[
            ["run_id", "global_step_canonical", "phase_id", "phase_name"] + tok_cols
        ].drop_duplicates()
        tok = tok.rename(columns=token_map)
        tok["token_row_present"] = True
        phase_fact = phase_fact.merge(
            tok,
            on=["run_id", "global_step_canonical", "phase_id", "phase_name"],
            how="left",
        )
    else:
        phase_fact["token_row_present"] = False

    shape_defaults = {
        "shape_gpu_power_mw_twa": None,
        "shape_gpu_power_mw_sample_mean": None,
        "shape_gpu_util_pct_twa": None,
        "shape_gpu_util_pct_sample_mean": None,
        "shape_sm_util_pct_twa": None,
        "shape_sm_util_pct_sample_mean": None,
        "shape_mem_util_pct_twa": None,
        "shape_mem_util_pct_sample_mean": None,
        "shape_sm_clock_mhz_twa": None,
        "shape_sm_clock_mhz_sample_mean": None,
        "shape_mem_clock_mhz_twa": None,
        "shape_mem_clock_mhz_sample_mean": None,
        "shape_temp_gpu_c_twa": None,
        "shape_temp_gpu_c_sample_mean": None,
        "shape_pcie_tx_bytes_s_twa": None,
        "shape_pcie_tx_bytes_s_sample_mean": None,
        "shape_pcie_rx_bytes_s_twa": None,
        "shape_pcie_rx_bytes_s_sample_mean": None,
        "shape_thr_sw_power_cap_frac_twa": None,
        "shape_thr_sw_power_cap_frac_sample_mean": None,
        "shape_thr_thermal_slowdown_frac_twa": None,
        "shape_thr_thermal_slowdown_frac_sample_mean": None,
        "shape_thr_hw_slowdown_frac_twa": None,
        "shape_thr_hw_slowdown_frac_sample_mean": None,
        "shape_thr_hw_power_brake_frac_twa": None,
        "shape_thr_hw_power_brake_frac_sample_mean": None,
        "shape_periodic_sample_count": 0,
    }
    if not hw_periodic_df.empty:
        p = hw_periodic_df.copy()
        p = p[p["source"] == "nvml"].copy()
        if not p.empty:
            p = filter_periodic_to_phase_window(p, phase_instances_df)
            p = add_time_weights(
                p,
                group_cols=["phase_instance_id", "source", "device_id"],
                ts_col="ts_monotonic_ns",
                out_col="sample_weight_ns",
            )

            numeric_metric_map = {
                "shape_gpu_power_mw": "gpu_power_mW",
                "shape_gpu_util_pct": "gpu_util_pct",
                "shape_sm_util_pct": "sm_util_pct",
                "shape_mem_util_pct": "mem_util_pct",
                "shape_sm_clock_mhz": "sm_clock_MHz",
                "shape_mem_clock_mhz": "mem_clock_MHz",
                "shape_temp_gpu_c": "temp_gpu_C",
                "shape_pcie_tx_bytes_s": "pcie_tx_bytes_s",
                "shape_pcie_rx_bytes_s": "pcie_rx_bytes_s",
            }
            bool_metric_map = {
                "shape_thr_sw_power_cap_frac": "thr_sw_power_cap",
                "shape_thr_thermal_slowdown_frac": "thr_thermal_slowdown",
                "shape_thr_hw_slowdown_frac": "thr_hw_slowdown",
                "shape_thr_hw_power_brake_frac": "thr_hw_power_brake",
            }

            shape_df = aggregate_periodic_metrics(
                p,
                group_cols=["phase_instance_id"],
                numeric_metric_map=numeric_metric_map,
                bool_metric_map=bool_metric_map,
                weight_col="sample_weight_ns",
                include_sample_mean=True,
                include_time_weighted_mean=True,
                sample_count_col="shape_periodic_sample_count",
            )
            phase_fact = phase_fact.merge(shape_df, on="phase_instance_id", how="left")
        else:
            for k, v in shape_defaults.items():
                phase_fact[k] = v
    else:
        for k, v in shape_defaults.items():
            phase_fact[k] = v

    if phase_outlier_flags_df.empty:
        phase_fact["is_outlier_sample"] = False
    else:
        phase_fact = phase_fact.merge(
            phase_outlier_flags_df[["phase_instance_id", "is_outlier_sample"]],
            on="phase_instance_id",
            how="left",
        )
        phase_fact["is_outlier_sample"] = phase_fact["is_outlier_sample"].fillna(False).astype(bool)

    phase_fact = _ensure_columns(phase_fact, PHASE_FACT_STRICT_COLUMNS)
    phase_fact = phase_fact[PHASE_FACT_STRICT_COLUMNS]

    validate_phase_fact(phase_fact)

    return phase_fact


def _coalesce_join_column(
    df: pd.DataFrame,
    target_col: str,
    *,
    default: Any = None,
    cast: Optional[str] = None,
) -> pd.DataFrame:
    """Coalesce merge suffix variants into target column, then drop suffix columns."""
    candidates = [target_col, f"{target_col}_y", f"{target_col}_x"]
    available = [c for c in candidates if c in df.columns]
    if not available:
        df[target_col] = default
        return df

    merged = pd.Series([None] * len(df), index=df.index)
    for c in available:
        merged = merged.where(merged.notna(), df[c])
    if default is not None:
        merged = merged.where(merged.notna(), default)

    if cast == "bool":
        merged = pd.Series(merged, index=df.index, dtype="boolean").fillna(False).astype(bool)
    elif cast == "int":
        merged = pd.to_numeric(merged, errors="coerce").fillna(0).astype(int)
    elif cast == "float":
        merged = pd.to_numeric(merged, errors="coerce")

    df[target_col] = merged
    for c in available:
        if c != target_col and c in df.columns:
            df = df.drop(columns=c)
    return df


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    out = n / d
    out = out.where(d > 0)
    return out


def _phase_selector_mask(phase_name_series: pd.Series, selector: str) -> pd.Series:
    s = phase_name_series.astype(str).str.lower()
    if selector == "rollout":
        return s.str.contains("rollout", na=False)
    if selector == "training":
        return s.str.contains("train", na=False)
    if selector == "validation":
        return s.str.contains("validation", na=False)
    return s == selector


def _recursive_find_first(obj: Any, keys: Iterable[str]) -> Optional[Any]:
    keyset = set(keys)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keyset and v is not None:
                return v
        for v in obj.values():
            out = _recursive_find_first(v, keyset)
            if out is not None:
                return out
    elif isinstance(obj, list):
        for item in obj:
            out = _recursive_find_first(item, keyset)
            if out is not None:
                return out
    return None


def _compute_lineage_root_map(runs_df: pd.DataFrame) -> Dict[str, Optional[str]]:
    if runs_df.empty:
        return {}

    parent_map: Dict[str, Optional[str]] = {}
    run_ids: set[str] = set()
    for row in runs_df.itertuples(index=False):
        run_id = getattr(row, "run_id", None)
        if run_id is None:
            continue
        run_ids.add(str(run_id))
        parent_map[str(run_id)] = getattr(row, "resume_parent_run_name", None)

    cache: Dict[str, Optional[str]] = {}

    def root_of(run_id: str) -> Optional[str]:
        if run_id in cache:
            return cache[run_id]
        seen: List[str] = []
        cur = run_id
        root: Optional[str] = run_id
        while True:
            if cur in cache:
                root = cache[cur]
                break
            if cur in seen:
                root = cur
                break
            seen.append(cur)
            parent = parent_map.get(cur)
            if not parent:
                root = cur
                break
            parent = str(parent)
            if parent not in run_ids:
                root = parent
                break
            cur = parent
        for item in seen:
            cache[item] = root
        return root

    out: Dict[str, Optional[str]] = {}
    for run_id in run_ids:
        out[run_id] = root_of(run_id)
    return out


def _parse_json_map(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str) and x:
        try:
            parsed = json.loads(x)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _sum_dict_values(d: Dict[str, Any]) -> int:
    total = 0
    for v in d.values():
        try:
            total += int(v)
        except Exception:
            continue
    return total


def _sanitize_phase_col(phase_name: Any) -> str:
    raw = "" if phase_name is None else str(phase_name).strip().lower()
    if not raw:
        return "unknown"
    return re.sub(r"[^0-9a-zA-Z]+", "_", raw).strip("_") or "unknown"


def _build_analysis_views(
    runs_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    phase_fact_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    hw_periodic_df: pd.DataFrame,
    hw_boundary_df: pd.DataFrame,
    ingestion_checks_df: pd.DataFrame,
    report_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Build analysis-ready views on top of canonical tables."""
    del lineage_df  # lineage is represented in runs_df/run_lineage already.

    run_identity_cols = [
        "run_id",
        "logical_run_group",
        "policy",
        "model",
        "dataset",
        "is_resumed_run",
        "resume_from_global_step",
        "run_name",
        "run_config_json",
        "slurm_partition",
        "slurm_nodes",
        "slurm_gpus_per_node",
    ]
    run_identity = _ensure_columns(runs_df.copy(), run_identity_cols)[run_identity_cols].drop_duplicates("run_id")
    run_identity["lineage_root_run_id"] = run_identity["run_id"]
    lineage_roots = _compute_lineage_root_map(run_identity)
    if lineage_roots:
        run_identity["lineage_root_run_id"] = run_identity["run_id"].map(lineage_roots).fillna(run_identity["run_id"])
    run_identity["is_checkpoint_continuation"] = run_identity["is_resumed_run"].fillna(False).astype(bool)
    run_identity["experiment_variant"] = run_identity["logical_run_group"]
    run_identity["variant_tags"] = run_identity["run_name"].fillna(run_identity["logical_run_group"])

    rollout_max_batched_tokens_vals: List[Optional[float]] = []
    enable_chunked_prefill_vals: List[Optional[bool]] = []
    for cfg in run_identity["run_config_json"].tolist():
        cfg_obj = _parse_json_map(cfg)
        val = _recursive_find_first(
            cfg_obj,
            keys=(
                "rollout_max_batched_tokens",
                "max_batched_tokens",
                "rollout_max_batch_tokens",
            ),
        )
        rollout_max_batched_tokens_vals.append(_safe_float(val))
        enable_chunked_prefill_vals.append(
            _safe_bool(_recursive_find_first(cfg_obj, keys=("enable_chunked_prefill", "rollout_enable_chunked_prefill")))
        )
    run_identity["rollout_max_batched_tokens"] = rollout_max_batched_tokens_vals
    run_identity["enable_chunked_prefill"] = enable_chunked_prefill_vals

    check_cols = [
        "run_id",
        "mismatch_count",
        "join_coverage_rate",
        "boundary_pair_integrity",
    ]
    checks = _ensure_columns(ingestion_checks_df.copy(), check_cols)[check_cols].drop_duplicates("run_id")
    checks["join_integrity_ok"] = (
        pd.to_numeric(checks["mismatch_count"], errors="coerce").fillna(0).astype(int) == 0
    ) & (
        pd.to_numeric(checks["join_coverage_rate"], errors="coerce").fillna(0.0) >= 1.0
    )
    checks["boundary_integrity_ok"] = (
        pd.to_numeric(checks["boundary_pair_integrity"], errors="coerce").fillna(0.0) >= 1.0
    )
    run_identity = run_identity.merge(
        checks[["run_id", "join_coverage_rate", "boundary_pair_integrity", "join_integrity_ok", "boundary_integrity_ok"]],
        on="run_id",
        how="left",
    )
    run_identity["join_integrity_ok"] = run_identity["join_integrity_ok"].fillna(False).astype(bool)
    run_identity["boundary_integrity_ok"] = run_identity["boundary_integrity_ok"].fillna(False).astype(bool)

    if phase_fact_df.empty:
        phase_fact_view = pd.DataFrame()
    else:
        phase_fact_view = phase_fact_df.copy()
        phase_fact_view = phase_fact_view.merge(
            run_identity[
                [
                    "run_id",
                    "lineage_root_run_id",
                    "logical_run_group",
                    "policy",
                    "model",
                    "dataset",
                    "resume_from_global_step",
                    "join_integrity_ok",
                    "boundary_integrity_ok",
                ]
            ],
            on="run_id",
            how="left",
            suffixes=("", "_run"),
        )
        if "logical_run_group_run" in phase_fact_view.columns:
            phase_fact_view["logical_run_group"] = phase_fact_view["logical_run_group"].fillna(
                phase_fact_view["logical_run_group_run"]
            )
            phase_fact_view = phase_fact_view.drop(columns=["logical_run_group_run"])

        step_num = pd.to_numeric(phase_fact_view["global_step_canonical"], errors="coerce")
        resume_num = pd.to_numeric(phase_fact_view["resume_from_global_step"], errors="coerce").fillna(0)
        phase_fact_view["absolute_global_step"] = (step_num + resume_num).where(step_num.notna())
        phase_fact_view["phase_time_s"] = pd.to_numeric(phase_fact_view["phase_duration_s_canonical"], errors="coerce")
        phase_fact_view["phase_start_ts"] = pd.to_numeric(phase_fact_view["phase_start_ts_monotonic_ns"], errors="coerce")
        phase_fact_view["phase_end_ts"] = pd.to_numeric(phase_fact_view["phase_end_ts_monotonic_ns"], errors="coerce")
        phase_fact_view["cpu_package_energy_j"] = pd.to_numeric(phase_fact_view["cpu_energy_j"], errors="coerce")
        phase_fact_view["dram_energy_j"] = pd.to_numeric(phase_fact_view["dram_energy_j"], errors="coerce")
        phase_fact_view["cpu_dram_energy_j"] = (
            phase_fact_view[["cpu_package_energy_j", "dram_energy_j"]]
            .fillna(0.0)
            .sum(axis=1)
            .where(phase_fact_view[["cpu_package_energy_j", "dram_energy_j"]].notna().any(axis=1))
        )
        phase_fact_view["total_energy_j"] = pd.to_numeric(phase_fact_view["total_energy_j"], errors="coerce")
        phase_fact_view["gpu_energy_j"] = pd.to_numeric(phase_fact_view["gpu_energy_j"], errors="coerce")

        phase_fact_view["rollout_output_tokens_total"] = pd.to_numeric(
            phase_fact_view.get("token_rollout_output_tokens_total"), errors="coerce"
        )
        phase_fact_view["rollout_total_tokens"] = pd.to_numeric(
            phase_fact_view.get("token_rollout_total_tokens"), errors="coerce"
        )
        phase_fact_view["train_tokens_effective_estimated"] = pd.to_numeric(
            phase_fact_view.get("token_train_tokens_effective_estimated"), errors="coerce"
        )

        rollout_mask = _phase_selector_mask(phase_fact_view["phase_name"], "rollout")
        training_mask = _phase_selector_mask(phase_fact_view["phase_name"], "training")

        phase_fact_view["j_per_output_token_rollout"] = _safe_divide(
            phase_fact_view["total_energy_j"],
            phase_fact_view["rollout_output_tokens_total"],
        ).where(rollout_mask)
        phase_fact_view["j_per_train_token_est"] = _safe_divide(
            phase_fact_view["total_energy_j"],
            phase_fact_view["train_tokens_effective_estimated"],
        ).where(training_mask)
        phase_fact_view["avg_power_w"] = _safe_divide(
            phase_fact_view["total_energy_j"],
            phase_fact_view["phase_time_s"],
        )

        phase_fact_view["gpu_util_mean"] = pd.to_numeric(phase_fact_view.get("shape_gpu_util_pct_twa"), errors="coerce")
        phase_fact_view["sm_util_mean"] = pd.to_numeric(phase_fact_view.get("shape_sm_util_pct_twa"), errors="coerce")
        phase_fact_view["mem_util_mean"] = pd.to_numeric(phase_fact_view.get("shape_mem_util_pct_twa"), errors="coerce")
        phase_fact_view["temp_gpu_mean"] = pd.to_numeric(phase_fact_view.get("shape_temp_gpu_c_twa"), errors="coerce")
        phase_fact_view["pcie_total_bytes_s_mean"] = (
            pd.to_numeric(phase_fact_view.get("shape_pcie_tx_bytes_s_twa"), errors="coerce").fillna(0.0)
            + pd.to_numeric(phase_fact_view.get("shape_pcie_rx_bytes_s_twa"), errors="coerce").fillna(0.0)
        ).where(
            pd.to_numeric(phase_fact_view.get("shape_pcie_tx_bytes_s_twa"), errors="coerce").notna()
            | pd.to_numeric(phase_fact_view.get("shape_pcie_rx_bytes_s_twa"), errors="coerce").notna()
        )

        phase_fact_view["throttle_sw_power_cap_rate"] = pd.to_numeric(
            phase_fact_view.get("shape_thr_sw_power_cap_frac_twa"), errors="coerce"
        )
        phase_fact_view["throttle_thermal_slowdown_rate"] = pd.to_numeric(
            phase_fact_view.get("shape_thr_thermal_slowdown_frac_twa"), errors="coerce"
        )
        phase_fact_view["throttle_hw_slowdown_rate"] = pd.to_numeric(
            phase_fact_view.get("shape_thr_hw_slowdown_frac_twa"), errors="coerce"
        )
        phase_fact_view["throttle_hw_power_brake_rate"] = pd.to_numeric(
            phase_fact_view.get("shape_thr_hw_power_brake_frac_twa"), errors="coerce"
        )

        # Step-level perf memory metrics are broadcast to each phase row for that run-step.
        phase_perf_cols = [
            "run_id",
            "global_step_canonical",
            "metric_perf_max_memory_allocated_gb",
            "metric_perf_max_memory_reserved_gb",
        ]
        phase_perf_df = _ensure_columns(wide_df.copy(), phase_perf_cols)[phase_perf_cols].rename(
            columns={
                "metric_perf_max_memory_allocated_gb": "max_memory_allocated_gb",
                "metric_perf_max_memory_reserved_gb": "max_memory_reserved_gb",
            }
        )
        for col in ["max_memory_allocated_gb", "max_memory_reserved_gb"]:
            phase_perf_df[col] = pd.to_numeric(phase_perf_df[col], errors="coerce")
        phase_fact_view = phase_fact_view.merge(
            phase_perf_df,
            on=["run_id", "global_step_canonical"],
            how="left",
        )

        step_group = ["run_id", "global_step_canonical"]
        step_energy_sum = phase_fact_view.groupby(step_group, dropna=False)["total_energy_j"].transform("sum")
        step_time_sum = phase_fact_view.groupby(step_group, dropna=False)["phase_time_s"].transform("sum")
        phase_fact_view["energy_share"] = _safe_divide(phase_fact_view["total_energy_j"], step_energy_sum)
        phase_fact_view["time_share"] = _safe_divide(phase_fact_view["phase_time_s"], step_time_sum)
        phase_fact_view["power_density_index"] = _safe_divide(
            phase_fact_view["energy_share"], phase_fact_view["time_share"]
        )

        phase_fact_view["join_integrity_ok"] = phase_fact_view["join_integrity_ok"].fillna(False).astype(bool)
        phase_fact_view["boundary_integrity_ok"] = phase_fact_view["boundary_integrity_ok"].fillna(False).astype(bool)

        phase_view_cols = [
            "run_id",
            "lineage_root_run_id",
            "logical_run_group",
            "policy",
            "model",
            "dataset",
            "global_step_canonical",
            "absolute_global_step",
            "phase_name",
            "phase_id",
            "phase_instance_id",
            "phase_time_s",
            "phase_start_ts",
            "phase_end_ts",
            "gpu_energy_j",
            "cpu_package_energy_j",
            "dram_energy_j",
            "cpu_dram_energy_j",
            "total_energy_j",
            "rollout_output_tokens_total",
            "rollout_total_tokens",
            "train_tokens_effective_estimated",
            "j_per_output_token_rollout",
            "j_per_train_token_est",
            "energy_share",
            "time_share",
            "power_density_index",
            "avg_power_w",
            "max_memory_allocated_gb",
            "max_memory_reserved_gb",
            "gpu_util_mean",
            "sm_util_mean",
            "mem_util_mean",
            "temp_gpu_mean",
            "pcie_total_bytes_s_mean",
            "throttle_sw_power_cap_rate",
            "throttle_thermal_slowdown_rate",
            "throttle_hw_slowdown_rate",
            "throttle_hw_power_brake_rate",
            "boundary_integrity_ok",
            "join_integrity_ok",
            "is_warmup_idle",
            "is_validation_step",
            "is_incomplete_phase",
            "is_outlier_sample",
        ]
        phase_fact_view = _ensure_columns(phase_fact_view, phase_view_cols)[phase_view_cols]

    if phase_fact_view.empty:
        step_fact_view = pd.DataFrame()
    else:
        pf = phase_fact_view.copy()
        rollout_mask = _phase_selector_mask(pf["phase_name"], "rollout")
        training_mask = _phase_selector_mask(pf["phase_name"], "training")
        group_cols = ["run_id", "global_step_canonical", "absolute_global_step"]

        pf["step_rollout_energy_j_component"] = pd.to_numeric(pf["total_energy_j"], errors="coerce").where(rollout_mask, 0.0)
        pf["step_train_energy_j_component"] = pd.to_numeric(pf["total_energy_j"], errors="coerce").where(training_mask, 0.0)
        pf["step_rollout_output_tokens_component"] = pd.to_numeric(
            pf["rollout_output_tokens_total"], errors="coerce"
        ).where(rollout_mask, 0.0)
        pf["step_rollout_total_tokens_component"] = pd.to_numeric(
            pf["rollout_total_tokens"], errors="coerce"
        ).where(rollout_mask, 0.0)
        pf["step_train_tokens_est_component"] = pd.to_numeric(
            pf["train_tokens_effective_estimated"], errors="coerce"
        ).where(training_mask, 0.0)

        step_fact_view = (
            pf.groupby(group_cols, dropna=False)
            .agg(
                step_time_s=("phase_time_s", "sum"),
                step_gpu_energy_j=("gpu_energy_j", "sum"),
                step_cpu_dram_energy_j=("cpu_dram_energy_j", "sum"),
                step_total_energy_j=("total_energy_j", "sum"),
                step_rollout_output_tokens=("step_rollout_output_tokens_component", "sum"),
                step_rollout_total_tokens=("step_rollout_total_tokens_component", "sum"),
                step_train_tokens_est=("step_train_tokens_est_component", "sum"),
                step_rollout_energy_j=("step_rollout_energy_j_component", "sum"),
                step_train_energy_j=("step_train_energy_j_component", "sum"),
                is_warmup_idle=("is_warmup_idle", "max"),
                is_validation_step=("is_validation_step", "max"),
                is_incomplete_phase=("is_incomplete_phase", "max"),
                is_outlier_sample=("is_outlier_sample", "max"),
            )
            .reset_index()
        )

        perf_cols = [
            "run_id",
            "global_step_canonical",
            "validation_logged",
            "metric_perf_throughput",
            "metric_perf_mfu_actor",
            "metric_perf_max_memory_allocated_gb",
            "metric_perf_max_memory_reserved_gb",
            "metric_rollout_straggler_ratio",
            "metric_rollout_sync_efficiency",
        ]
        perf_df = _ensure_columns(wide_df.copy(), perf_cols)[perf_cols]
        perf_df = perf_df.rename(
            columns={
                "metric_perf_throughput": "throughput_tokens_s",
                "metric_perf_mfu_actor": "mfu_actor",
                "metric_perf_max_memory_allocated_gb": "max_memory_allocated_gb",
                "metric_perf_max_memory_reserved_gb": "max_memory_reserved_gb",
                "metric_rollout_straggler_ratio": "straggler_ratio",
                "metric_rollout_sync_efficiency": "sync_efficiency",
            }
        )
        for col in [
            "throughput_tokens_s",
            "mfu_actor",
            "max_memory_allocated_gb",
            "max_memory_reserved_gb",
            "straggler_ratio",
            "sync_efficiency",
        ]:
            perf_df[col] = pd.to_numeric(perf_df[col], errors="coerce")

        step_fact_view = step_fact_view.merge(
            perf_df,
            on=["run_id", "global_step_canonical"],
            how="left",
        )

        step_fact_view["validation_logged"] = (
            step_fact_view["validation_logged"].fillna(step_fact_view["is_validation_step"]).fillna(False).astype(bool)
        )
        step_fact_view["step_j_per_output_token"] = _safe_divide(
            step_fact_view["step_total_energy_j"], step_fact_view["step_rollout_output_tokens"]
        )
        step_fact_view["rollout_j_per_output_token"] = _safe_divide(
            step_fact_view["step_rollout_energy_j"], step_fact_view["step_rollout_output_tokens"]
        )
        step_fact_view["train_j_per_effective_token"] = _safe_divide(
            step_fact_view["step_train_energy_j"], step_fact_view["step_train_tokens_est"]
        )
        step_fact_view["step_edp"] = (
            pd.to_numeric(step_fact_view["step_total_energy_j"], errors="coerce")
            * pd.to_numeric(step_fact_view["step_time_s"], errors="coerce")
        )

        step_fact_view = step_fact_view.merge(
            run_identity[
                ["run_id", "lineage_root_run_id", "policy", "model", "variant_tags", "join_integrity_ok", "boundary_integrity_ok"]
            ],
            on="run_id",
            how="left",
        )
        step_fact_view["join_integrity_ok"] = step_fact_view["join_integrity_ok"].fillna(False).astype(bool)
        step_fact_view["boundary_integrity_ok"] = step_fact_view["boundary_integrity_ok"].fillna(False).astype(bool)

        step_cols = [
            "run_id",
            "lineage_root_run_id",
            "policy",
            "model",
            "variant_tags",
            "global_step_canonical",
            "absolute_global_step",
            "validation_logged",
            "step_time_s",
            "step_gpu_energy_j",
            "step_cpu_dram_energy_j",
            "step_total_energy_j",
            "step_rollout_output_tokens",
            "step_rollout_total_tokens",
            "step_train_tokens_est",
            "throughput_tokens_s",
            "mfu_actor",
            "max_memory_allocated_gb",
            "max_memory_reserved_gb",
            "straggler_ratio",
            "sync_efficiency",
            "step_j_per_output_token",
            "rollout_j_per_output_token",
            "train_j_per_effective_token",
            "step_edp",
            "join_integrity_ok",
            "boundary_integrity_ok",
            "is_warmup_idle",
            "is_validation_step",
            "is_incomplete_phase",
            "is_outlier_sample",
        ]
        step_fact_view = _ensure_columns(step_fact_view, step_cols)[step_cols]

    if run_identity.empty:
        run_summary_view = pd.DataFrame()
    else:
        run_summary_view = run_identity[
            [
                "run_id",
                "lineage_root_run_id",
                "logical_run_group",
                "policy",
                "model",
                "dataset",
                "experiment_variant",
                "variant_tags",
                "rollout_max_batched_tokens",
                "enable_chunked_prefill",
                "is_checkpoint_continuation",
                "slurm_partition",
                "slurm_nodes",
                "slurm_gpus_per_node",
                "join_coverage_rate",
                "boundary_pair_integrity",
            ]
        ].copy()
        run_summary_view = run_summary_view.rename(
            columns={
                "boundary_pair_integrity": "phase_boundary_integrity_rate",
            }
        )

        if not step_fact_view.empty:
            step_aggs = (
                step_fact_view.groupby("run_id", dropna=False)
                .agg(
                    num_steps_included=("global_step_canonical", "nunique"),
                    mean_step_j_per_output_token=("step_j_per_output_token", "mean"),
                    median_step_j_per_output_token=("step_j_per_output_token", "median"),
                    mean_rollout_j_per_output_token=("rollout_j_per_output_token", "mean"),
                    mean_train_j_per_effective_token=("train_j_per_effective_token", "mean"),
                    mean_max_memory_allocated_gb=("max_memory_allocated_gb", "mean"),
                    max_max_memory_allocated_gb=("max_memory_allocated_gb", "max"),
                    mean_max_memory_reserved_gb=("max_memory_reserved_gb", "mean"),
                    max_max_memory_reserved_gb=("max_memory_reserved_gb", "max"),
                )
                .reset_index()
            )
            run_summary_view = run_summary_view.merge(step_aggs, on="run_id", how="left")
        else:
            run_summary_view["num_steps_included"] = 0

        if not phase_fact_view.empty:
            phase_aggs = (
                phase_fact_view.groupby("run_id", dropna=False)
                .agg(
                    num_phases_included=("phase_instance_id", "nunique"),
                    mean_phase_power_density_index=("power_density_index", "mean"),
                    mean_throttle_sw_power_cap_rate=("throttle_sw_power_cap_rate", "mean"),
                    mean_throttle_thermal_slowdown_rate=("throttle_thermal_slowdown_rate", "mean"),
                    mean_throttle_hw_slowdown_rate=("throttle_hw_slowdown_rate", "mean"),
                    mean_throttle_hw_power_brake_rate=("throttle_hw_power_brake_rate", "mean"),
                    mean_pcie_total_bytes_s=("pcie_total_bytes_s_mean", "mean"),
                )
                .reset_index()
            )
            run_summary_view = run_summary_view.merge(phase_aggs, on="run_id", how="left")

            phase_pdi = (
                phase_fact_view.groupby(["run_id", "phase_name"], dropna=False)["power_density_index"]
                .mean()
                .reset_index()
            )
            if not phase_pdi.empty:
                phase_pdi["phase_name"] = phase_pdi["phase_name"].apply(_sanitize_phase_col)
                phase_pdi = phase_pdi.pivot_table(
                    index="run_id",
                    columns="phase_name",
                    values="power_density_index",
                    aggfunc="mean",
                ).reset_index()
                phase_pdi.columns = [
                    "run_id" if c == "run_id" else f"mean_power_density_index_phase_{c}"
                    for c in phase_pdi.columns
                ]
                run_summary_view = run_summary_view.merge(phase_pdi, on="run_id", how="left")
        else:
            run_summary_view["num_phases_included"] = 0

        val_col = "metric_val_core_openai_gsm8k_reward_mean_1"
        val_df = _ensure_columns(wide_df.copy(), ["run_id", "global_step_canonical", val_col])[
            ["run_id", "global_step_canonical", val_col]
        ]
        val_df[val_col] = pd.to_numeric(val_df[val_col], errors="coerce")
        val_rows: List[Dict[str, Any]] = []
        step_energy_map = (
            step_fact_view[["run_id", "global_step_canonical", "step_total_energy_j"]].copy()
            if not step_fact_view.empty
            else pd.DataFrame(columns=["run_id", "global_step_canonical", "step_total_energy_j"])
        )
        for run_id, grp in val_df.groupby("run_id", dropna=False):
            g = grp.sort_values("global_step_canonical")
            valid = g[g[val_col].notna()]
            final_metric = valid[val_col].iloc[-1] if not valid.empty else None
            best_metric = valid[val_col].max() if not valid.empty else None
            best_step = None
            if not valid.empty:
                best_row = valid.loc[valid[val_col].idxmax()]
                best_step = best_row["global_step_canonical"]
            energy_to_best = None
            if best_step is not None and not step_energy_map.empty:
                em = step_energy_map[step_energy_map["run_id"] == run_id].copy()
                em = em.sort_values("global_step_canonical")
                em["cum_energy"] = pd.to_numeric(em["step_total_energy_j"], errors="coerce").fillna(0.0).cumsum()
                hit = em[em["global_step_canonical"] <= best_step]
                if not hit.empty:
                    energy_to_best = float(hit["cum_energy"].iloc[-1])
            val_rows.append(
                {
                    "run_id": run_id,
                    "final_validation_metric": final_metric,
                    "best_validation_metric": best_metric,
                    "energy_to_best_validation_j": energy_to_best,
                }
            )
        val_summary = pd.DataFrame(val_rows)
        run_summary_view = run_summary_view.merge(val_summary, on="run_id", how="left")

        for col in ["num_steps_included", "num_phases_included"]:
            run_summary_view[col] = pd.to_numeric(run_summary_view[col], errors="coerce").fillna(0).astype(int)

    if step_fact_view.empty:
        comparison_view = pd.DataFrame()
    else:
        run_group_info = run_identity[
            [
                "run_id",
                "policy",
                "model",
                "experiment_variant",
                "rollout_max_batched_tokens",
                "enable_chunked_prefill",
                "is_checkpoint_continuation",
            ]
        ].drop_duplicates("run_id")
        sf = step_fact_view.merge(run_group_info, on=["run_id", "policy", "model"], how="left")
        group_keys = [
            "policy",
            "model",
            "experiment_variant",
            "rollout_max_batched_tokens",
            "enable_chunked_prefill",
            "is_checkpoint_continuation",
        ]
        comparison_view = (
            sf.groupby(group_keys, dropna=False)
            .agg(
                n_runs=("run_id", "nunique"),
                n_steps=("global_step_canonical", "count"),
                overall_j_per_output_token_mean=("step_j_per_output_token", "mean"),
                overall_j_per_output_token_median=("step_j_per_output_token", "median"),
                rollout_j_per_output_token_mean=("rollout_j_per_output_token", "mean"),
                train_j_per_effective_token_mean=("train_j_per_effective_token", "mean"),
                straggler_ratio_mean=("straggler_ratio", "mean"),
                sync_efficiency_mean=("sync_efficiency", "mean"),
                mfu_actor_mean=("mfu_actor", "mean"),
            )
            .reset_index()
        )

        if not phase_fact_view.empty:
            pf_group = phase_fact_view.merge(run_group_info, on=["run_id", "policy", "model"], how="left")
            phase_shares = (
                pf_group.groupby(group_keys + ["phase_name"], dropna=False)["energy_share"]
                .mean()
                .reset_index()
            )
            if not phase_shares.empty:
                phase_shares["phase_name"] = phase_shares["phase_name"].apply(_sanitize_phase_col)
                phase_shares = phase_shares.pivot_table(
                    index=group_keys,
                    columns="phase_name",
                    values="energy_share",
                    aggfunc="mean",
                ).reset_index()
                phase_shares.columns = [
                    c if isinstance(c, str) and c in group_keys else f"phase_energy_share_mean_{c}"
                    for c in phase_shares.columns
                ]
                comparison_view = comparison_view.merge(phase_shares, on=group_keys, how="left")

            coupling_rows: List[Dict[str, Any]] = []
            for keys, grp in pf_group.groupby(group_keys, dropna=False):
                u = pd.to_numeric(grp["gpu_util_mean"], errors="coerce")
                pwr = pd.to_numeric(grp["avg_power_w"], errors="coerce")
                mask = u.notna() & pwr.notna()
                slope = None
                corr = None
                if int(mask.sum()) >= 2:
                    uu = u[mask]
                    pp = pwr[mask]
                    var_u = float(((uu - uu.mean()) ** 2).mean())
                    if var_u > 0:
                        slope = float((((uu - uu.mean()) * (pp - pp.mean())).mean()) / var_u)
                    corr = float(uu.corr(pp))
                row = {k: v for k, v in zip(group_keys, keys if isinstance(keys, tuple) else (keys,))}
                row["util_power_slope_b"] = slope
                row["util_power_corr"] = corr
                coupling_rows.append(row)
            coupling_df = pd.DataFrame(coupling_rows)
            comparison_view = comparison_view.merge(coupling_df, on=group_keys, how="left")

    if hw_periodic_df.empty:
        device_timeseries_view = pd.DataFrame()
    else:
        device_timeseries_view = hw_periodic_df.copy()
        device_timeseries_view["gpu_power_mw"] = pd.to_numeric(device_timeseries_view.get("gpu_power_mW"), errors="coerce")
        device_timeseries_view["power_w"] = device_timeseries_view["gpu_power_mw"] / 1000.0
        device_timeseries_view["gpu_util_pct"] = pd.to_numeric(device_timeseries_view.get("gpu_util_pct"), errors="coerce")
        device_timeseries_view["sm_util_pct"] = pd.to_numeric(device_timeseries_view.get("sm_util_pct"), errors="coerce")
        device_timeseries_view["mem_util_pct"] = pd.to_numeric(device_timeseries_view.get("mem_util_pct"), errors="coerce")
        device_timeseries_view["sm_clock_mhz"] = pd.to_numeric(device_timeseries_view.get("sm_clock_MHz"), errors="coerce")
        device_timeseries_view["mem_clock_mhz"] = pd.to_numeric(device_timeseries_view.get("mem_clock_MHz"), errors="coerce")
        device_timeseries_view["temp_gpu_c"] = pd.to_numeric(device_timeseries_view.get("temp_gpu_C"), errors="coerce")
        device_timeseries_view["pcie_tx_bytes_s"] = pd.to_numeric(device_timeseries_view.get("pcie_tx_bytes_s"), errors="coerce")
        device_timeseries_view["pcie_rx_bytes_s"] = pd.to_numeric(device_timeseries_view.get("pcie_rx_bytes_s"), errors="coerce")
        device_timeseries_view["pcie_total_bytes_s"] = (
            device_timeseries_view["pcie_tx_bytes_s"].fillna(0.0)
            + device_timeseries_view["pcie_rx_bytes_s"].fillna(0.0)
        ).where(
            device_timeseries_view["pcie_tx_bytes_s"].notna() | device_timeseries_view["pcie_rx_bytes_s"].notna()
        )
        for c in ["thr_sw_power_cap", "thr_thermal_slowdown", "thr_hw_slowdown", "thr_hw_power_brake"]:
            raw_col = device_timeseries_view.get(c)
            bool_col = pd.Series(raw_col, index=device_timeseries_view.index, dtype="boolean")
            device_timeseries_view[c] = bool_col.fillna(False).astype(bool)

        device_cols = [
            "run_id",
            "global_step_canonical",
            "phase_name",
            "phase_id",
            "device_kind",
            "device_id",
            "source",
            "ts_monotonic_ns",
            "elapsed_seconds",
            "power_w",
            "gpu_power_mw",
            "gpu_util_pct",
            "sm_util_pct",
            "mem_util_pct",
            "sm_clock_mhz",
            "mem_clock_mhz",
            "temp_gpu_c",
            "pcie_tx_bytes_s",
            "pcie_rx_bytes_s",
            "pcie_total_bytes_s",
            "thr_sw_power_cap",
            "thr_thermal_slowdown",
            "thr_hw_slowdown",
            "thr_hw_power_brake",
        ]
        device_timeseries_view = _ensure_columns(device_timeseries_view, device_cols)[device_cols]

    report_cols = [
        "run_id",
        "included",
        "reason",
        "parse_error_files",
        "analysis_mask_is_warmup_idle_count",
        "analysis_mask_is_validation_step_count",
        "analysis_mask_is_incomplete_phase_count",
        "analysis_mask_is_outlier_sample_count",
    ]
    report_local = _ensure_columns(report_df.copy(), report_cols)[report_cols]
    report_local["parse_error_count"] = report_local["parse_error_files"].apply(
        lambda x: _sum_dict_values(_parse_json_map(x))
    )
    report_local["excluded_reason"] = report_local["reason"].where(~report_local["included"].astype(bool), None)

    if hw_boundary_df.empty:
        neg_df = pd.DataFrame(columns=["run_id", "negative_delta_count"])
    else:
        hb = hw_boundary_df.copy()
        gpu_neg = pd.to_numeric(hb.get("phase_gpu_energy_delta_J"), errors="coerce") < 0
        rapl_neg = pd.to_numeric(hb.get("phase_domain_energy_delta_j"), errors="coerce") < 0
        hb["negative_delta_row"] = gpu_neg.fillna(False) | rapl_neg.fillna(False)
        neg_df = (
            hb.groupby("run_id", dropna=False)["negative_delta_row"].sum().reset_index(name="negative_delta_count")
        )

    integrity_view = run_identity[
        ["run_id", "join_coverage_rate", "boundary_pair_integrity"]
    ].rename(
        columns={
            "join_coverage_rate": "step_join_coverage",
            "boundary_pair_integrity": "phase_boundary_integrity_rate",
        }
    )
    integrity_view = integrity_view.merge(
        neg_df,
        on="run_id",
        how="left",
    )
    integrity_view = integrity_view.merge(
        report_local[
            [
                "run_id",
                "parse_error_count",
                "excluded_reason",
                "analysis_mask_is_warmup_idle_count",
                "analysis_mask_is_validation_step_count",
                "analysis_mask_is_incomplete_phase_count",
                "analysis_mask_is_outlier_sample_count",
            ]
        ],
        on="run_id",
        how="left",
    )
    integrity_view["negative_delta_count"] = pd.to_numeric(integrity_view["negative_delta_count"], errors="coerce").fillna(0).astype(int)
    integrity_view["parse_error_count"] = pd.to_numeric(integrity_view["parse_error_count"], errors="coerce").fillna(0).astype(int)
    for c in [
        "analysis_mask_is_warmup_idle_count",
        "analysis_mask_is_validation_step_count",
        "analysis_mask_is_incomplete_phase_count",
        "analysis_mask_is_outlier_sample_count",
    ]:
        integrity_view[c] = pd.to_numeric(integrity_view[c], errors="coerce").fillna(0).astype(int)
    integrity_view["integrity_ok"] = (
        (pd.to_numeric(integrity_view["step_join_coverage"], errors="coerce").fillna(0.0) >= 1.0)
        & (pd.to_numeric(integrity_view["phase_boundary_integrity_rate"], errors="coerce").fillna(0.0) >= 1.0)
        & (integrity_view["negative_delta_count"] == 0)
        & (integrity_view["parse_error_count"] == 0)
        & integrity_view["excluded_reason"].isna()
    )

    return {
        "phase_fact_view": phase_fact_view,
        "step_fact_view": step_fact_view,
        "run_summary_view": run_summary_view,
        "comparison_view": comparison_view,
        "device_timeseries_view": device_timeseries_view,
        "integrity_view": integrity_view,
    }
