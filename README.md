# verl-profiler

`verl-profiler` is a Slurm workflow for researchers who want to run a profiled veRL training job and then ingest the resulting monitoring outputs into analysis tables.

The repository currently ships one small validation config:

- [`profiling_scripts/experiments/qwen_test`](profiling_scripts/experiments/qwen_test)

It runs:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- policy: `ppo`
- partition: `gpu`
- nodes: `1`
- gpus per node: `1`
- total training steps: `1`

## Before You Submit

Your repo checkout and the Python environment used by the launcher must both live on a filesystem that is visible from the login node and the compute nodes.

The launcher expects a Python environment at `./verl-env` by default. From the repository root:

```bash
python3.10 -m venv verl-env
source verl-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

If your cluster uses a different Python executable, use that instead of `python3.10`, but keep the environment name `verl-env`, or set `VENV_ACTIVATE` explicitly to the activation script you want the launcher to source.

If you already have a shared environment outside this repo, point the launcher at it:

```bash
export VENV_ACTIVATE=/path/to/shared/verl-env/bin/activate
source "$VENV_ACTIVATE"
```

Validate the environment before submitting:

```bash
source verl-env/bin/activate
python -c "import ray, verl, torch"
which ray
ray --version
```

`profiling_scripts/token.sh` is optional. The launcher checks for it to support gated model access, but the shipped `qwen_test` config does not require it.

## Required Runtime Paths

Real launches require these environment variables:

- `SCRATCH_DIR`: shared writable scratch space for datasets, checkpoints, and transient monitoring outputs
- `ARCHIVE_RESULTS_ROOT`: destination root for migrated monitoring results
- `ARCHIVE_LOGS_ROOT`: destination root for archived logs

Example:

```bash
export SCRATCH_DIR=/path/to/shared/scratch
export ARCHIVE_RESULTS_ROOT=/path/to/archive/results
export ARCHIVE_LOGS_ROOT=/path/to/archive/logs
export VENV_ACTIVATE=/path/to/shared/verl-env/bin/activate
```

The shipped `qwen_test` config contains explicit `output` and `error` paths in [`profiling_scripts/experiments/qwen_test/slurm.json`](profiling_scripts/experiments/qwen_test/slurm.json). By default they are relative Slurm log paths (`slurm-%A_%a.out` and `slurm-%A_%a.err`), which write beside the submit directory. If your cluster expects logs somewhere else, edit those two fields before you submit.

## Quick Start

Local preflight:

```bash
export SCRATCH_DIR=/path/to/shared/scratch
export ARCHIVE_RESULTS_ROOT=/path/to/archive/results
export ARCHIVE_LOGS_ROOT=/path/to/archive/logs
export VENV_ACTIVATE=/path/to/shared/verl-env/bin/activate
bash profiling_scripts/submit_sweep.sh profiling_scripts/experiments/qwen_test --check
```

Smoke-style preflight:

```bash
bash profiling_scripts/submit_sweep.sh profiling_scripts/experiments/qwen_test --smoke --check
```

Real Slurm submission:

```bash
bash profiling_scripts/submit_sweep.sh profiling_scripts/experiments/qwen_test
```

## Expected Outputs

During and after a successful run, the launcher records:

- Slurm stdout/stderr at the `output` and `error` locations from `slurm.json`
- migrated monitoring artifacts under `$ARCHIVE_RESULTS_ROOT`
- archived logs under `$ARCHIVE_LOGS_ROOT`
- run metadata such as `run_config.json`, `slurm_config.json`, and `slurm_job_ids.txt`

## Troubleshooting

If a compute node cannot see the environment or the environment is incomplete, the launcher will now fail early with explicit messages. Two common failures from the real `qwen_test` trial were:

- `ERROR: VENV_ACTIVATE not found: ...`
- `ERROR: ray not found after sourcing VENV_ACTIVATE=...`

If you see either:

1. confirm the environment path is visible from compute nodes
2. activate it manually and run `which ray`
3. rerun `python -c "import ray, verl, torch"` inside that environment

The launcher also requires explicit path configuration. If `SCRATCH_DIR`, `ARCHIVE_RESULTS_ROOT`, `ARCHIVE_LOGS_ROOT`, `slurm.json.output`, or `slurm.json.error` are missing, `submit_sweep.sh` now exits before submission with a clear error.

## Build Analysis Datasets

After a successful run, ingest the archived results with:

```bash
python3 analysis_scripts/build_run_datasets.py \
  --results-root "$ARCHIVE_RESULTS_ROOT" \
  --output-root /path/to/output/datasets \
  --overwrite
```

Make sure `--results-root` points at the same archive results root used by the launcher.

## Where to Edit Things

- Change training/model settings in [`profiling_scripts/experiments/qwen_test/runs.json`](profiling_scripts/experiments/qwen_test/runs.json)
- Change Slurm partition/resources/log paths in [`profiling_scripts/experiments/qwen_test/slurm.json`](profiling_scripts/experiments/qwen_test/slurm.json)
- Launch with [`profiling_scripts/submit_sweep.sh`](profiling_scripts/submit_sweep.sh)
