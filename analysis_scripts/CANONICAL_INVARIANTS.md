# Canonical Invariants

This contract defines the canonical rules every downstream table/view must follow.

1. **Energy + time source**
   - Use boundary logs only (`nvml_boundary.jsonl`, `rapl_boundary.jsonl`).
   - Use explicit `START`/`END` phase events and monotonic energy counters.
   - Do **not** reconstruct phase energy by integrating sampled power.

2. **Shape metrics source**
   - Use periodic logs only (`nvml_periodic.jsonl`, `rapl_periodic.jsonl`).
   - Metrics include utilization, clocks, temperatures, throttling, PCIe.
   - Aggregate over each phase window as time-weighted averages.

3. **Token denominator source**
   - Use `tokens_and_steps.jsonl` only.
   - Do **not** reconstruct token denominators from other artifacts.

## Canonical Keys

- `run_id` is the canonical run key.
- `global_step` is the canonical step key.
- Canonical phase key: `run_id + global_step + phase_name + phase_id`.

## Intended Usage

- Any derived table with phase energy/time must trace back to boundary logs.
- Any derived table with shape/hardware behavior must trace back to periodic logs.
- Any token-normalized metric must trace token denominators to `tokens_and_steps.jsonl`.
