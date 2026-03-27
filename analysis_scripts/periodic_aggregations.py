"""Reusable periodic metric aggregation utilities.

Implements monotonic phase-window filtering and time-weighted aggregations.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import pandas as pd


def filter_periodic_to_phase_window(
    periodic_df: pd.DataFrame,
    phase_instances_df: pd.DataFrame,
    phase_instance_col: str = "phase_instance_id",
    ts_col: str = "ts_monotonic_ns",
    start_col: str = "phase_start_ts_monotonic_ns",
    end_col: str = "phase_end_ts_monotonic_ns",
) -> pd.DataFrame:
    """Filter periodic samples to their phase windows using monotonic timestamps."""
    if periodic_df.empty:
        return periodic_df.copy()

    if phase_instance_col not in periodic_df.columns:
        return periodic_df.copy()

    win = phase_instances_df[[phase_instance_col, start_col, end_col]].drop_duplicates()
    d = periodic_df.merge(win, on=phase_instance_col, how="left")

    ts = pd.to_numeric(d[ts_col], errors="coerce")
    start = pd.to_numeric(d[start_col], errors="coerce")
    end = pd.to_numeric(d[end_col], errors="coerce")

    return d[(ts >= start) & (ts <= end)].copy()


def add_time_weights(
    periodic_df: pd.DataFrame,
    group_cols: Sequence[str],
    ts_col: str = "ts_monotonic_ns",
    out_col: str = "sample_weight_ns",
) -> pd.DataFrame:
    """Add adjacent-delta time weights per monotonic timeline group."""
    if periodic_df.empty:
        d = periodic_df.copy()
        d[out_col] = pd.Series(dtype="float64")
        return d

    d = periodic_df.sort_values(list(group_cols) + [ts_col]).copy()
    grp = d.groupby(list(group_cols), dropna=False)
    d["_next_ts"] = grp[ts_col].shift(-1)
    d[out_col] = pd.to_numeric(d["_next_ts"] - d[ts_col], errors="coerce")

    fallback = grp[out_col].transform(lambda s: s[s > 0].median())
    d[out_col] = d[out_col].fillna(fallback).fillna(1.0).clip(lower=1.0)
    d.drop(columns=["_next_ts"], inplace=True)
    return d


def _weighted_mean(values: pd.Series, weights: pd.Series) -> Optional[float]:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return None
    denom = float(w[mask].sum())
    if denom <= 0:
        return None
    return float((v[mask] * w[mask]).sum() / denom)


def aggregate_periodic_metrics(
    periodic_df: pd.DataFrame,
    group_cols: Sequence[str],
    numeric_metric_map: Dict[str, str],
    bool_metric_map: Optional[Dict[str, str]] = None,
    weight_col: str = "sample_weight_ns",
    include_sample_mean: bool = True,
    include_time_weighted_mean: bool = True,
    sample_count_col: str = "shape_periodic_sample_count",
) -> pd.DataFrame:
    """Aggregate periodic metrics per group with sample mean and/or time-weighted mean.

    metric maps use output base names as keys, e.g. {"shape_gpu_power_mw": "gpu_power_mW"}.
    Output columns are suffixed with `_sample_mean` and `_twa`.
    """
    if bool_metric_map is None:
        bool_metric_map = {}

    cols = list(group_cols)
    if periodic_df.empty:
        return pd.DataFrame(columns=cols + [sample_count_col])

    rows = []
    for keys, grp in periodic_df.groupby(cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(cols, keys)}
        row[sample_count_col] = int(len(grp))

        weights = grp[weight_col] if weight_col in grp.columns else pd.Series([1.0] * len(grp), index=grp.index)

        for out_base, src_col in numeric_metric_map.items():
            if src_col in grp.columns:
                src = pd.to_numeric(grp[src_col], errors="coerce")
                if include_sample_mean:
                    row[f"{out_base}_sample_mean"] = float(src.mean()) if src.notna().any() else None
                if include_time_weighted_mean:
                    row[f"{out_base}_twa"] = _weighted_mean(src, weights)
            else:
                if include_sample_mean:
                    row[f"{out_base}_sample_mean"] = None
                if include_time_weighted_mean:
                    row[f"{out_base}_twa"] = None

        for out_base, src_col in bool_metric_map.items():
            if src_col in grp.columns:
                src = grp[src_col].astype(float)
                if include_sample_mean:
                    row[f"{out_base}_sample_mean"] = float(src.mean()) if src.notna().any() else None
                if include_time_weighted_mean:
                    row[f"{out_base}_twa"] = _weighted_mean(src, weights)
            else:
                if include_sample_mean:
                    row[f"{out_base}_sample_mean"] = None
                if include_time_weighted_mean:
                    row[f"{out_base}_twa"] = None

        rows.append(row)

    return pd.DataFrame(rows)
