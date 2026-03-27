"""Validation and integrity checks for dataset ingestion.

This module isolates all check logic from data manipulation code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def build_phase_outlier_flags(hw_boundary_df: pd.DataFrame) -> pd.DataFrame:
    """Detect outlier samples per phase instance from boundary rows."""
    if hw_boundary_df.empty:
        return pd.DataFrame(columns=["phase_instance_id", "is_outlier_sample"])

    b = hw_boundary_df.copy()
    b["phase_gpu_energy_delta_J"] = pd.to_numeric(b.get("phase_gpu_energy_delta_J"), errors="coerce")
    b["phase_domain_energy_delta_j"] = pd.to_numeric(b.get("phase_domain_energy_delta_j"), errors="coerce")
    b["phase_duration_s"] = pd.to_numeric(b.get("phase_duration_s"), errors="coerce")
    b["gpu_energy_mJ"] = pd.to_numeric(b.get("gpu_energy_mJ"), errors="coerce")
    b["cpu_energy_uJ"] = pd.to_numeric(b.get("cpu_energy_uJ"), errors="coerce")

    b["row_outlier"] = (
        (b["phase_gpu_energy_delta_J"] < 0)
        | (b["phase_domain_energy_delta_j"] < 0)
        | (b["phase_duration_s"] < 0)
    )

    counter_reset_pairs = set()
    if "boundary_pair_key" in b.columns:
        for pair_key, grp in b.groupby("boundary_pair_key", dropna=False):
            if pair_key is None:
                continue
            starts = grp[grp["phase_event"] == "START"]
            ends = grp[grp["phase_event"] == "END"]
            if len(starts) != 1 or len(ends) != 1:
                continue
            s = starts.iloc[0]
            e = ends.iloc[0]
            if s.get("source") == "nvml":
                if pd.notna(s.get("gpu_energy_mJ")) and pd.notna(e.get("gpu_energy_mJ")) and e.get("gpu_energy_mJ") < s.get("gpu_energy_mJ"):
                    counter_reset_pairs.add(pair_key)
            if s.get("source") == "rapl":
                if pd.notna(s.get("cpu_energy_uJ")) and pd.notna(e.get("cpu_energy_uJ")) and e.get("cpu_energy_uJ") < s.get("cpu_energy_uJ"):
                    counter_reset_pairs.add(pair_key)

    b["counter_reset_outlier"] = b["boundary_pair_key"].isin(counter_reset_pairs)
    b["is_outlier_sample"] = b["row_outlier"] | b["counter_reset_outlier"]

    out = (
        b.groupby("phase_instance_id", dropna=False)["is_outlier_sample"]
        .max()
        .reset_index()
    )
    out["is_outlier_sample"] = out["is_outlier_sample"].fillna(False).astype(bool)
    return out


def validate_phase_fact(phase_fact_df: pd.DataFrame) -> None:
    """Validate one-row-per-phase-instance constraint."""
    if phase_fact_df.empty:
        return
    if phase_fact_df["phase_instance_id"].duplicated().any():
        dup_ids = phase_fact_df.loc[phase_fact_df["phase_instance_id"].duplicated(), "phase_instance_id"].head(10).tolist()
        raise RuntimeError(f"phase_fact must be one row per phase_instance_id; duplicates found: {dup_ids}")


def build_step_index_and_checks(
    run_id: str,
    run_step_observations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build step_index_map rows and per-run step/join checks."""
    if not run_step_observations:
        return [], {
            "run_id": run_id,
            "raw_step_distinct_count": 0,
            "raw_iteration_distinct_count": 0,
            "overlap_distinct_count": 0,
            "raw_step_only_count": 0,
            "raw_iteration_only_count": 0,
            "mismatch_count": 0,
            "observation_count": 0,
            "mismatch_rate": 0.0,
            "join_coverage_rate": 1.0,
        }

    df = pd.DataFrame(run_step_observations)
    index_df = (
        df.groupby(["run_id", "raw_step", "raw_iteration", "canonical_step", "mismatch_flag"], dropna=False)
        .size()
        .reset_index(name="observation_count")
    )

    raw_step_values = set(df["raw_step"].dropna().astype(int).tolist())
    raw_iteration_values = set(df["raw_iteration"].dropna().astype(int).tolist())
    overlap_steps = raw_step_values.intersection(raw_iteration_values)

    mismatch_count = int(df["mismatch_flag"].astype(bool).sum())
    obs_count = int(len(df))

    checks = {
        "run_id": run_id,
        "raw_step_distinct_count": len(raw_step_values),
        "raw_iteration_distinct_count": len(raw_iteration_values),
        "overlap_distinct_count": len(overlap_steps),
        "raw_step_only_count": len(raw_step_values - raw_iteration_values),
        "raw_iteration_only_count": len(raw_iteration_values - raw_step_values),
        "mismatch_count": mismatch_count,
        "observation_count": obs_count,
        "mismatch_rate": (mismatch_count / float(obs_count)) if obs_count else 0.0,
        "join_coverage_rate": (len(overlap_steps) / float(len(raw_step_values))) if raw_step_values else 1.0,
    }

    return index_df.to_dict(orient="records"), checks


def build_boundary_integrity_outputs(
    run_id: str,
    run_boundary_rows_local: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float, int, int, bool, int]:
    """Build phase-instance timestamps and boundary pair integrity rows."""
    if not run_boundary_rows_local:
        return [], [], 1.0, 0, 0, False, 0

    run_boundary_df = pd.DataFrame(run_boundary_rows_local)
    step_series = pd.to_numeric(run_boundary_df.get("global_step_canonical"), errors="coerce")
    max_step = int(step_series.max()) if not step_series.dropna().empty else None

    phase_rows_local: List[Dict[str, Any]] = []
    for (phase_instance_id, phase_name, phase_id, step_val), grp in run_boundary_df.groupby(
        ["phase_instance_id", "phase_name", "phase_id", "global_step_canonical"], dropna=False
    ):
        start_ts = pd.to_numeric(
            grp.loc[grp.get("phase_event") == "START", "ts_monotonic_ns"],
            errors="coerce",
        )
        end_ts = pd.to_numeric(
            grp.loc[grp.get("phase_event") == "END", "ts_monotonic_ns"],
            errors="coerce",
        )
        phase_rows_local.append(
            {
                "run_id": run_id,
                "global_step": step_val,
                "global_step_canonical": step_val,
                "global_step_raw_stepfield": None,
                "global_step_raw_iterationfield": step_val,
                "phase_name": phase_name,
                "phase_id": phase_id,
                "phase_instance_id": phase_instance_id,
                "phase_start_ts_monotonic_ns": int(start_ts.min()) if not start_ts.dropna().empty else None,
                "phase_end_ts_monotonic_ns": int(end_ts.max()) if not end_ts.dropna().empty else None,
                "start_row_count": int((grp.get("phase_event") == "START").sum()),
                "end_row_count": int((grp.get("phase_event") == "END").sum()),
                "boundary_row_count": int(len(grp)),
            }
        )

    pair_rows_local: List[Dict[str, Any]] = []
    invalid_terminal_start_only_count = 0
    invalid_pair_count = 0
    for (pair_key, phase_instance_id, source, device_id), grp in run_boundary_df.groupby(
        ["boundary_pair_key", "phase_instance_id", "source", "device_id"], dropna=False
    ):
        start_count = int((grp.get("phase_event") == "START").sum())
        end_count = int((grp.get("phase_event") == "END").sum())
        is_valid = start_count == 1 and end_count == 1
        pair_step_series = pd.to_numeric(grp.get("global_step_canonical"), errors="coerce")
        pair_step = int(pair_step_series.iloc[0]) if not pair_step_series.dropna().empty else None
        is_terminal_start_only_pair = (
            not is_valid and start_count == 1 and end_count == 0 and max_step is not None and pair_step == max_step
        )
        if not is_valid:
            invalid_pair_count += 1
            if is_terminal_start_only_pair:
                invalid_terminal_start_only_count += 1
        pair_rows_local.append(
            {
                "run_id": run_id,
                "boundary_pair_key": pair_key,
                "phase_instance_id": phase_instance_id,
                "source": source,
                "device_id": device_id,
                "global_step_canonical": pair_step,
                "start_count": start_count,
                "end_count": end_count,
                "row_count": int(len(grp)),
                "is_valid_pair": bool(is_valid),
                "is_terminal_start_only_pair": bool(is_terminal_start_only_pair),
            }
        )

    boundary_pair_total = len(pair_rows_local)
    boundary_pair_valid = sum(1 for row in pair_rows_local if row["is_valid_pair"])
    boundary_pair_integrity = (
        boundary_pair_valid / float(boundary_pair_total) if boundary_pair_total else 1.0
    )
    terminal_start_only_exception = invalid_pair_count > 0 and invalid_pair_count == invalid_terminal_start_only_count

    return (
        phase_rows_local,
        pair_rows_local,
        boundary_pair_integrity,
        boundary_pair_total,
        boundary_pair_valid,
        terminal_start_only_exception,
        invalid_terminal_start_only_count,
    )


def validate_ingestion_checks(ingestion_checks_df: pd.DataFrame) -> Tuple[int, float]:
    """Hard-fail checks shared by ingestion entrypoints.

    Raises RuntimeError if mismatch or boundary integrity constraints are violated.
    """
    total_mismatch_count = int(
        pd.to_numeric(ingestion_checks_df.get("mismatch_count"), errors="coerce").fillna(0).sum()
    )
    if total_mismatch_count > 0:
        bad_runs = ingestion_checks_df[
            pd.to_numeric(ingestion_checks_df.get("mismatch_count"), errors="coerce") > 0
        ][["run_id", "mismatch_count", "mismatch_rate"]]
        raise RuntimeError(
            "Detected non-zero step mismatch count; refusing to write datasets. "
            f"Run-level mismatches: {bad_runs.to_dict(orient='records')}"
        )

    min_boundary_pair_integrity = (
        float(pd.to_numeric(ingestion_checks_df.get("boundary_pair_integrity"), errors="coerce").min())
        if not ingestion_checks_df.empty
        else 1.0
    )
    if min_boundary_pair_integrity < 1.0:
        exception_mask = (
            pd.Series(False, index=ingestion_checks_df.index)
            if "terminal_start_only_boundary_exception" not in ingestion_checks_df.columns
            else ingestion_checks_df["terminal_start_only_boundary_exception"].fillna(False).astype(bool)
        )
        bad_integrity_runs = ingestion_checks_df[
            (pd.to_numeric(ingestion_checks_df.get("boundary_pair_integrity"), errors="coerce") < 1.0) & ~exception_mask
        ][["run_id", "boundary_pair_integrity", "boundary_pair_total", "boundary_pair_valid"]]
        if not bad_integrity_runs.empty:
            raise RuntimeError(
                "Detected boundary pair integrity < 1.0; refusing to write datasets. "
                f"Run-level boundary integrity failures: {bad_integrity_runs.to_dict(orient='records')}"
            )

    return total_mismatch_count, min_boundary_pair_integrity
