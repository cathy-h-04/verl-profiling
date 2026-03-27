# Analysis Scripts

This directory exposes one supported user-facing entrypoint:

```bash
python3 analysis_scripts/build_run_datasets.py --results-root ... --output-root ... [--include-subdir ...] [--overwrite]
```

Use a Python 3.7+ interpreter for this script.

The script ingests completed run directories that contain profiler artifacts such as:

- `experiment_name.txt`
- `slurm_job_ids.txt`
- `run_config.json`
- `slurm_config.json`
- telemetry JSONL files such as `nvml_boundary.jsonl`, `nvml_periodic.jsonl`, `rapl_boundary.jsonl`, `rapl_periodic.jsonl`, and `tokens_and_steps.jsonl`

## Typical Usage

Ingest the default results tree:

```bash
python3 analysis_scripts/build_run_datasets.py \
  --results-root /path/to/archive/results \
  --output-root /path/to/output/datasets \
  --overwrite
```

Ingest a subset of runs:

```bash
python3 analysis_scripts/build_run_datasets.py \
  --results-root /path/to/archive/results \
  --include-subdir /path/to/archive/results/monitoring_val \
  --output-root /path/to/output/datasets \
  --overwrite
```

## Outputs

`build_run_datasets.py` writes normalized Parquet tables under `--output-root`, including:

- `runs.parquet`
- `run_lineage.parquet`
- `ingestion_checks.parquet`
- `phase_fact.parquet`
- `hardware_boundary.parquet`
- `hardware_periodic.parquet`
- analysis-ready view tables such as `phase_fact_view.parquet`, `step_fact_view.parquet`, and `run_summary_view.parquet`

## Files Kept In This Directory

- [`build_run_datasets.py`](build_run_datasets.py): the supported CLI
- [`utils.py`](utils.py): shared ingestion helpers
- [`ingestion_checks.py`](ingestion_checks.py): invariant and integrity checks
- [`periodic_aggregations.py`](periodic_aggregations.py): time-window and periodic aggregation helpers
- [`CANONICAL_INVARIANTS.md`](CANONICAL_INVARIANTS.md): ingestion contract notes
