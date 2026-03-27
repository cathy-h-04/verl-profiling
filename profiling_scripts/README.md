# Profiling Scripts

This directory contains the Slurm launcher used by the repository.

## Environment

By default the launcher expects a repo-local activation script at `./verl-env/bin/activate`.

Create it from the repository root:

```bash
python3.10 -m venv verl-env
source verl-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

If you already have a shared environment elsewhere, export it before launch:

```bash
export VENV_ACTIVATE=/path/to/shared/verl-env/bin/activate
source "$VENV_ACTIVATE"
```

Check that the environment is usable before submitting:

```bash
source verl-env/bin/activate
python -c "import ray, verl, torch"
which ray
ray --version
```

`profiling_scripts/token.sh` is optional and is only used for gated model access.

## Required Environment Variables

Set these before launch:

```bash
export SCRATCH_DIR=/path/to/shared/scratch
export ARCHIVE_RESULTS_ROOT=/path/to/archive/results
export ARCHIVE_LOGS_ROOT=/path/to/archive/logs
export VENV_ACTIVATE=/path/to/shared/verl-env/bin/activate
```

## Shipped Config

The only shipped config is:

- [`profiling_scripts/experiments/qwen_test`](experiments/qwen_test)

Its `slurm.json` uses explicit `output` and `error` paths. By default they are relative paths in the submit directory. Edit them if your cluster requires a different log location.

## Launch Commands

Local preflight:

```bash
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

If `SCRATCH_DIR`, `ARCHIVE_RESULTS_ROOT`, `ARCHIVE_LOGS_ROOT`, `slurm.json.output`, `slurm.json.error`, or `VENV_ACTIVATE` are missing, the launcher now exits before submission with a clear error.
