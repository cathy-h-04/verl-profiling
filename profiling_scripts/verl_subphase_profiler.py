#!/usr/bin/env python3
"""
Phase + sub-phase profiler for verl RLHF training.
Provides IPC between controller and monitoring script, with optional sub-phase timing logs.
"""

import atexit
import json
import os
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

PhaseType = Literal["idle", "rollout", "rl_policy", "training", "validation", "other"]

PHASE_IDS = {
    "idle": 0,
    "rollout": 1,
    "rl_policy": 2,
    "training": 3,
    "validation": 4,
    "other": 5,
}


def _parse_int_env(var_names: List[str]) -> Optional[int]:
    for name in var_names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def _parse_float_env(var_names: List[str]) -> Optional[float]:
    for name in var_names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _require_env_str(name: str) -> str:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        raise RuntimeError(f"{name} must be set before using verl_subphase_profiler.")
    return str(raw)


class PhaseProfiler:
    """Writer class - used by verl trainer to signal phase transitions."""

    def __init__(self, experiment_name: str, enable: bool = True):
        self.experiment_name = experiment_name
        self.enabled = enable
        self.current_phase: PhaseType = "idle"
        self.current_iteration = 0
        self.phase_start_time = None
        self.granularity = "phase"
        self._cleanup_done = False
        self._phase_open = False

        self._jsonl_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._periodic_thread = None

        self._phase_start_monotonic_ns: Dict[str, int] = {}
        self._phase_start_nvml_energy_mj: Dict[str, Dict[str, Optional[int]]] = {}
        self._phase_start_rapl_energy_uj: Dict[str, Dict[str, Optional[int]]] = {}
        self._cpu_stat_prev: Optional[Tuple[int, int]] = None

        # Runtime topology metadata
        self.node = socket.gethostname()
        self.pid = os.getpid()
        self.rank = _parse_int_env(["RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"])
        self.local_rank = _parse_int_env(["LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"])
        self.world_size = _parse_int_env(["WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"])
        self._run_start_monotonic_ns = time.monotonic_ns()
        # Default periodic sampling cadence: 2 Hz.
        self.sample_interval_s = 0.5
        sample_interval = _parse_float_env(["VERL_TELEMETRY_SAMPLE_INTERVAL_S"])
        sample_hz = _parse_float_env(["VERL_TELEMETRY_SAMPLE_HZ"])
        if sample_hz is not None and sample_hz > 0:
            sample_interval = 1.0 / sample_hz
        if sample_interval is not None and sample_interval > 0:
            # Guardrail: avoid pathological intervals by default.
            self.sample_interval_s = max(0.1, sample_interval)

        if not self.enabled:
            return

        # Use file-based IPC in monitoring directory
        scratch_dir = _require_env_str("SCRATCH_DIR")
        monitoring_dir = (
            os.environ.get("MONITORING_DIR")
            or os.environ.get("VERL_FILE_LOGGER_ROOT")
            or f"{scratch_dir}/logs"
        )
        self.state_dir = Path(monitoring_dir)
        self.data_dir = (
            Path(os.environ["MONITORING_DIR"])
            if os.environ.get("MONITORING_DIR")
            else Path(monitoring_dir) / experiment_name
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.state_dir / f"phase_state_{experiment_name}.json"
        self.nvml_boundary_file = self.data_dir / "nvml_boundary.jsonl"
        self.nvml_periodic_file = self.data_dir / "nvml_periodic.jsonl"
        self.rapl_boundary_file = self.data_dir / "rapl_boundary.jsonl"
        self.rapl_periodic_file = self.data_dir / "rapl_periodic.jsonl"
        self.tokens_file = self.data_dir / "tokens_and_steps.jsonl"

        self._nvml = None
        self._nvml_driver_version = None
        self._nvml_devices: List[Dict[str, Any]] = []
        self._rapl_domains: List[Dict[str, Any]] = []

        # Initialize with idle state
        self._write_state(
            {
                "phase_id": PHASE_IDS["idle"],
                "phase_name": "idle",
                "iteration": 0,
                "timestamp": time.time(),
            }
        )
        self._init_nvml()
        self._init_rapl_domains()
        self._start_periodic_sampler()
        atexit.register(self.cleanup)
        print(f"✓ Phase profiler initialized: {self.state_file}")

    def _write_state(self, state: Dict):
        """Write state to file atomically."""
        if not self.enabled:
            return
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(state, f)
        temp_file.replace(self.state_file)

    def _json_default(self, value: Any):
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)

    def _append_jsonl(self, path: Path, record: Dict[str, Any]):
        if not self.enabled:
            return
        with self._jsonl_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=self._json_default) + "\n")

    def _base_record(
        self,
        *,
        phase_name: str,
        record_type: str,
        phase_event: Optional[str] = None,
        iter_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        rec: Dict[str, Any] = {
            "ts_wall_ms": int(time.time() * 1000),
            "ts_monotonic_ns": time.monotonic_ns(),
            "timestamp": time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()),
            "elapsed_seconds": max(0.0, (time.monotonic_ns() - self._run_start_monotonic_ns) / 1_000_000_000.0),
            "node": self.node,
            "pid": self.pid,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "phase_name": phase_name,
            "phase_id": PHASE_IDS.get(phase_name, PHASE_IDS["other"]),
            "record_type": record_type,
            "iteration": iter_id,
        }
        if phase_event is not None:
            rec["phase_event"] = phase_event
        return rec

    def _safe_call(self, field: str, fn, error_fields: List[str]):
        try:
            return fn()
        except Exception:
            error_fields.append(field)
            return None

    def _start_periodic_sampler(self):
        self._periodic_thread = threading.Thread(
            target=self._periodic_sampler_loop,
            name=f"phase_profiler_periodic_{self.experiment_name}",
            daemon=True,
        )
        self._periodic_thread.start()

    def _periodic_sampler_loop(self):
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_tick:
                self._stop_event.wait(next_tick - now)
                if self._stop_event.is_set():
                    break
            try:
                self._emit_nvml_periodic_snapshot()
            except Exception as e:
                print(f"Warning: NVML periodic snapshot failed: {e}")
            try:
                self._emit_rapl_periodic_snapshot()
            except Exception as e:
                print(f"Warning: RAPL periodic snapshot failed: {e}")
            # Anchor to a monotonic deadline and avoid immediate catch-up bursts.
            # If delayed beyond one interval, reset cadence from "now".
            next_tick += self.sample_interval_s
            now = time.monotonic()
            if now > next_tick + self.sample_interval_s:
                next_tick = now + self.sample_interval_s

    def _init_nvml(self):
        try:
            import pynvml  # type: ignore
        except Exception:
            return

        try:
            pynvml.nvmlInit()
        except Exception:
            return

        self._nvml = pynvml
        try:
            raw_driver = pynvml.nvmlSystemGetDriverVersion()
            self._nvml_driver_version = raw_driver.decode("utf-8") if isinstance(raw_driver, bytes) else raw_driver
        except Exception:
            self._nvml_driver_version = None

        try:
            device_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            device_count = 0
        visible_indices = self._resolve_visible_nvml_indices(device_count)
        for gpu_index in visible_indices:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                continue
            gpu_uuid = None
            gpu_name = None
            try:
                raw_uuid = pynvml.nvmlDeviceGetUUID(handle)
                gpu_uuid = raw_uuid.decode("utf-8") if isinstance(raw_uuid, bytes) else raw_uuid
            except Exception:
                pass
            try:
                raw_name = pynvml.nvmlDeviceGetName(handle)
                gpu_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else raw_name
            except Exception:
                pass
            self._nvml_devices.append(
                {
                    "gpu_index": gpu_index,
                    "gpu_uuid": gpu_uuid,
                    "gpu_name": gpu_name,
                    "handle": handle,
                }
            )

    def _resolve_visible_nvml_indices(self, device_count: int) -> List[int]:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not cvd:
            return list(range(device_count))

        tokens = [t for t in re.split(r"[,\s]+", cvd) if t]
        indices: List[int] = []
        uuid_tokens: List[str] = []

        for token in tokens:
            if re.fullmatch(r"[0-9]+", token):
                idx = int(token)
                if 0 <= idx < device_count:
                    indices.append(idx)
            else:
                uuid_tokens.append(token.lower())

        if uuid_tokens and self._nvml is not None:
            uuid_to_index: Dict[str, int] = {}
            for idx in range(device_count):
                try:
                    handle = self._nvml.nvmlDeviceGetHandleByIndex(idx)
                    raw_uuid = self._nvml.nvmlDeviceGetUUID(handle)
                    uuid_str = raw_uuid.decode("utf-8") if isinstance(raw_uuid, bytes) else raw_uuid
                    if uuid_str:
                        uuid_to_index[uuid_str.lower()] = idx
                except Exception:
                    continue
            for token in uuid_tokens:
                for uuid_str, idx in uuid_to_index.items():
                    if uuid_str == token or uuid_str.startswith(token):
                        indices.append(idx)
                        break

        # Preserve order but deduplicate.
        deduped = list(dict.fromkeys(indices))
        return deduped if deduped else list(range(device_count))

    def _read_int_file(self, path: Path) -> int:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())

    def _read_text_file(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _sort_rapl_path_key(self, path: Path):
        nums = [int(x) for x in path.name.split(":")[1:]]
        return nums

    def _init_rapl_domains(self):
        base = Path("/sys/class/powercap")
        if not base.exists():
            return

        socket_paths = sorted(
            [p for p in base.glob("intel-rapl:[0-9]*") if p.is_dir() and p.name.count(":") == 1],
            key=self._sort_rapl_path_key,
        )

        for socket_path in socket_paths:
            try:
                socket_idx = int(socket_path.name.split(":")[1])
            except Exception:
                continue
            self._add_rapl_domain(socket_path, socket_idx)
            child_paths = sorted(
                [p for p in socket_path.glob(f"{socket_path.name}:*") if p.is_dir()],
                key=self._sort_rapl_path_key,
            )
            for child in child_paths:
                self._add_rapl_domain(child, socket_idx)

    def _add_rapl_domain(self, domain_path: Path, socket_idx: int):
        error_fields: List[str] = []
        domain_name = self._safe_call(
            "domain_name",
            lambda: self._read_text_file(domain_path / "name"),
            error_fields,
        )
        max_range = self._safe_call(
            "max_energy_range_uJ",
            lambda: self._read_int_file(domain_path / "max_energy_range_uj"),
            error_fields,
        )
        safe_name = re.sub(r"[^a-z0-9_]+", "_", (domain_name or "unknown").lower())
        if safe_name.startswith("package"):
            rapl_domain = f"package-{socket_idx}"
        elif safe_name.startswith("dram"):
            rapl_domain = f"dram-{socket_idx}"
        else:
            rapl_domain = f"{safe_name}-{socket_idx}"

        self._rapl_domains.append(
            {
                "domain_id": domain_path.name,
                "domain_path": str(domain_path),
                "domain_name": domain_name,
                "rapl_domain": rapl_domain,
                "rapl_socket": f"socket-{socket_idx}",
                "max_energy_range_uJ": max_range,
                "init_error_fields": error_fields,
            }
        )

    def _nvml_const(self, *names: str):
        if self._nvml is None:
            return None
        for name in names:
            if hasattr(self._nvml, name):
                return getattr(self._nvml, name)
        return None

    def _decode_throttle_reasons(self, raw_value: Optional[int]) -> Dict[str, Optional[bool]]:
        if raw_value is None:
            return {
                "thr_sw_power_cap": None,
                "thr_hw_power_brake": None,
                "thr_thermal_slowdown": None,
                "thr_hw_slowdown": None,
                "thr_sync_boost": None,
                "thr_idle": None,
                "thr_sw_thermal_slowdown": None,
                "thr_hw_thermal_slowdown": None,
                "thr_apps_clocks_setting": None,
                "thr_display_clock_setting": None,
            }

        def _flag(const_name: Optional[int]) -> Optional[bool]:
            if const_name is None:
                return None
            return bool(int(raw_value) & int(const_name))

        sw_power_cap = _flag(
            self._nvml_const("nvmlClocksThrottleReasonSwPowerCap", "nvmlClocksEventReasonSwPowerCap")
        )
        hw_power_brake = _flag(
            self._nvml_const(
                "nvmlClocksThrottleReasonHwPowerBrakeSlowdown", "nvmlClocksEventReasonHwPowerBrakeSlowdown"
            )
        )
        sw_thermal = _flag(
            self._nvml_const("nvmlClocksThrottleReasonSwThermalSlowdown", "nvmlClocksEventReasonSwThermalSlowdown")
        )
        hw_thermal = _flag(
            self._nvml_const("nvmlClocksThrottleReasonHwThermalSlowdown", "nvmlClocksEventReasonHwThermalSlowdown")
        )
        thermal_slowdown = None
        if sw_thermal is not None or hw_thermal is not None:
            thermal_slowdown = bool(sw_thermal or hw_thermal)

        return {
            "thr_sw_power_cap": sw_power_cap,
            "thr_hw_power_brake": hw_power_brake,
            "thr_thermal_slowdown": thermal_slowdown,
            "thr_hw_slowdown": _flag(
                self._nvml_const("nvmlClocksThrottleReasonHwSlowdown", "nvmlClocksEventReasonHwSlowdown")
            ),
            "thr_sync_boost": _flag(
                self._nvml_const("nvmlClocksThrottleReasonSyncBoost", "nvmlClocksEventReasonSyncBoost")
            ),
            "thr_idle": _flag(self._nvml_const("nvmlClocksThrottleReasonGpuIdle", "nvmlClocksEventReasonGpuIdle")),
            "thr_sw_thermal_slowdown": sw_thermal,
            "thr_hw_thermal_slowdown": hw_thermal,
            "thr_apps_clocks_setting": _flag(
                self._nvml_const(
                    "nvmlClocksThrottleReasonApplicationsClocksSetting",
                    "nvmlClocksEventReasonApplicationsClocksSetting",
                )
            ),
            "thr_display_clock_setting": _flag(
                self._nvml_const(
                    "nvmlClocksThrottleReasonDisplayClockSetting",
                    "nvmlClocksEventReasonDisplayClockSetting",
                )
            ),
        }

    def _sample_nvml_gpu(self, gpu: Dict[str, Any], include_identity: bool) -> Dict[str, Any]:
        if self._nvml is None:
            return {}

        h = gpu["handle"]
        error_fields: List[str] = []
        util = self._safe_call("utilization", lambda: self._nvml.nvmlDeviceGetUtilizationRates(h), error_fields)
        mem_info = self._safe_call("memory_info", lambda: self._nvml.nvmlDeviceGetMemoryInfo(h), error_fields)
        throttle_raw = self._safe_call(
            "clocks_throttle_reasons_raw",
            lambda: int(self._nvml.nvmlDeviceGetCurrentClocksThrottleReasons(h)),
            error_fields,
        )

        tx_counter = self._nvml_const("NVML_PCIE_UTIL_TX_BYTES")
        rx_counter = self._nvml_const("NVML_PCIE_UTIL_RX_BYTES")
        tx_kib_s = (
            self._safe_call(
                "pcie_tx_bytes_s",
                lambda: int(self._nvml.nvmlDeviceGetPcieThroughput(h, tx_counter)),
                error_fields,
            )
            if tx_counter is not None
            else None
        )
        rx_kib_s = (
            self._safe_call(
                "pcie_rx_bytes_s",
                lambda: int(self._nvml.nvmlDeviceGetPcieThroughput(h, rx_counter)),
                error_fields,
            )
            if rx_counter is not None
            else None
        )

        sample: Dict[str, Any] = {
            "gpu_energy_mJ": self._safe_call(
                "gpu_energy_mJ",
                lambda: int(self._nvml.nvmlDeviceGetTotalEnergyConsumption(h)),
                error_fields,
            ),
            "gpu_power_mW": self._safe_call("gpu_power_mW", lambda: int(self._nvml.nvmlDeviceGetPowerUsage(h)), error_fields),
            "gpu_power_limit_mW": self._safe_call(
                "gpu_power_limit_mW",
                lambda: int(self._nvml.nvmlDeviceGetPowerManagementLimit(h)),
                error_fields,
            ),
            "gpu_enforced_power_limit_mW": self._safe_call(
                "gpu_enforced_power_limit_mW",
                lambda: int(self._nvml.nvmlDeviceGetEnforcedPowerLimit(h)),
                error_fields,
            ),
            "gpu_util_pct": getattr(util, "gpu", None) if util is not None else None,
            # NVML "gpu" utilization is closest to SM busy percentage.
            "sm_util_pct": getattr(util, "gpu", None) if util is not None else None,
            "mem_util_pct": getattr(util, "memory", None) if util is not None else None,
            "sm_clock_MHz": self._safe_call(
                "sm_clock_MHz",
                lambda: int(self._nvml.nvmlDeviceGetClockInfo(h, self._nvml_const("NVML_CLOCK_SM"))),
                error_fields,
            ),
            "mem_clock_MHz": self._safe_call(
                "mem_clock_MHz",
                lambda: int(self._nvml.nvmlDeviceGetClockInfo(h, self._nvml_const("NVML_CLOCK_MEM"))),
                error_fields,
            ),
            "graphics_clock_MHz": self._safe_call(
                "graphics_clock_MHz",
                lambda: int(self._nvml.nvmlDeviceGetClockInfo(h, self._nvml_const("NVML_CLOCK_GRAPHICS"))),
                error_fields,
            ),
            "pstate": self._safe_call("pstate", lambda: int(self._nvml.nvmlDeviceGetPerformanceState(h)), error_fields),
            "temp_gpu_C": self._safe_call(
                "temp_gpu_C",
                lambda: int(self._nvml.nvmlDeviceGetTemperature(h, self._nvml_const("NVML_TEMPERATURE_GPU"))),
                error_fields,
            ),
            "temp_mem_C": self._safe_call(
                "temp_mem_C",
                lambda: int(self._nvml.nvmlDeviceGetTemperature(h, self._nvml_const("NVML_TEMPERATURE_MEMORY"))),
                error_fields,
            ),
            "mem_total_B": getattr(mem_info, "total", None) if mem_info is not None else None,
            "mem_used_B": getattr(mem_info, "used", None) if mem_info is not None else None,
            "mem_free_B": getattr(mem_info, "free", None) if mem_info is not None else None,
            "clocks_throttle_reasons_raw": throttle_raw,
            "pcie_tx_bytes_s": int(tx_kib_s * 1024) if tx_kib_s is not None else None,
            "pcie_rx_bytes_s": int(rx_kib_s * 1024) if rx_kib_s is not None else None,
            "pcie_link_gen": self._safe_call(
                "pcie_link_gen",
                lambda: int(self._nvml.nvmlDeviceGetCurrPcieLinkGeneration(h)),
                error_fields,
            ),
            "pcie_link_width": self._safe_call(
                "pcie_link_width",
                lambda: int(self._nvml.nvmlDeviceGetCurrPcieLinkWidth(h)),
                error_fields,
            ),
        }
        sample.update(self._decode_throttle_reasons(throttle_raw))
        if include_identity:
            sample.update(
                {
                    "gpu_index": gpu.get("gpu_index"),
                    "gpu_uuid": gpu.get("gpu_uuid"),
                    "gpu_name": gpu.get("gpu_name"),
                    "driver_version": self._nvml_driver_version,
                }
            )
        sample["error_fields"] = error_fields
        return sample

    def _phase_duration_s(self, phase_name: str) -> Optional[float]:
        start_ns = self._phase_start_monotonic_ns.get(phase_name)
        if start_ns is None:
            return None
        return max(0.0, (time.monotonic_ns() - start_ns) / 1_000_000_000.0)

    def _emit_nvml_boundary_snapshot(self, phase_name: str, phase_event: str, iteration: Optional[int]):
        if not self._nvml_devices:
            return

        phase_duration_s = self._phase_duration_s(phase_name) if phase_event == "END" else None
        energy_map: Dict[str, Optional[int]] = {}
        start_map = self._phase_start_nvml_energy_mj.get(phase_name, {})

        for gpu in self._nvml_devices:
            sample = self._sample_nvml_gpu(gpu, include_identity=True)
            gpu_key = sample.get("gpu_uuid") or f"gpu-index-{sample.get('gpu_index')}"
            energy_map[gpu_key] = sample.get("gpu_energy_mJ")

            rec = self._base_record(
                phase_name=phase_name,
                record_type="PHASE_BOUNDARY",
                phase_event=phase_event,
                iter_id=iteration,
            )
            rec.update(sample)
            if phase_duration_s is not None:
                rec["phase_duration_s"] = phase_duration_s
            if phase_event == "END":
                start_energy = start_map.get(gpu_key)
                end_energy = sample.get("gpu_energy_mJ")
                if start_energy is None or end_energy is None:
                    rec["phase_gpu_energy_delta_J"] = None
                else:
                    delta_mj = end_energy - start_energy
                    rec["phase_gpu_energy_delta_J"] = delta_mj / 1000.0
            self._append_jsonl(self.nvml_boundary_file, rec)

        if phase_event == "START":
            self._phase_start_nvml_energy_mj[phase_name] = energy_map

    def _emit_nvml_periodic_snapshot(self):
        if not self._nvml_devices:
            return
        phase_name = self.current_phase
        iteration = self.current_iteration
        for gpu in self._nvml_devices:
            sample = self._sample_nvml_gpu(gpu, include_identity=True)
            rec = self._base_record(
                phase_name=phase_name,
                record_type="PERIODIC",
                iter_id=iteration,
            )
            rec.update(sample)
            self._append_jsonl(self.nvml_periodic_file, rec)

    def _emit_rapl_boundary_snapshot(self, phase_name: str, phase_event: str, iteration: Optional[int]):
        if not self._rapl_domains:
            return

        phase_duration_s = self._phase_duration_s(phase_name) if phase_event == "END" else None
        start_map = self._phase_start_rapl_energy_uj.get(phase_name, {})
        current_map: Dict[str, Optional[int]] = {}

        for domain in self._rapl_domains:
            domain_path = Path(domain["domain_path"])
            error_fields = list(domain.get("init_error_fields", []))
            energy_uj = self._safe_call(
                "cpu_energy_uJ",
                lambda: self._read_int_file(domain_path / "energy_uj"),
                error_fields,
            )
            current_map[domain["domain_id"]] = energy_uj

            rec = self._base_record(
                phase_name=phase_name,
                record_type="PHASE_BOUNDARY",
                phase_event=phase_event,
                iter_id=iteration,
            )
            rec.update(
                {
                    "cpu_energy_uJ": energy_uj,
                    "max_energy_range_uJ": domain.get("max_energy_range_uJ"),
                    "rapl_domain": domain.get("rapl_domain"),
                    "domain_path": domain.get("domain_path"),
                }
            )
            if phase_duration_s is not None:
                rec["phase_duration_s"] = phase_duration_s

            if phase_event == "END":
                start_energy = start_map.get(domain["domain_id"])
                delta_uj = None
                if start_energy is not None and energy_uj is not None:
                    delta_uj = energy_uj - start_energy
                    if delta_uj < 0:
                        max_range = domain.get("max_energy_range_uJ")
                        if isinstance(max_range, int) and max_range > 0:
                            delta_uj += max_range
                        else:
                            error_fields.append("phase_domain_energy_delta_uJ_wrap_without_max_range")
                            delta_uj = None
                rec["phase_domain_energy_delta_uJ"] = delta_uj

            rec["error_fields"] = error_fields
            self._append_jsonl(self.rapl_boundary_file, rec)

        if phase_event == "START":
            self._phase_start_rapl_energy_uj[phase_name] = current_map

    def _read_cpu_util_pct_total(self) -> Optional[float]:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as f:
                first = f.readline().strip()
        except Exception:
            return None
        if not first.startswith("cpu "):
            return None
        parts = first.split()
        values = [int(v) for v in parts[1:]]
        if len(values) < 5:
            return None
        idle = values[3] + values[4]  # idle + iowait
        total = sum(values)
        if self._cpu_stat_prev is None:
            self._cpu_stat_prev = (total, idle)
            return None
        prev_total, prev_idle = self._cpu_stat_prev
        self._cpu_stat_prev = (total, idle)
        delta_total = total - prev_total
        delta_idle = idle - prev_idle
        if delta_total <= 0:
            return None
        return 100.0 * (1.0 - float(delta_idle) / float(delta_total))

    def _read_loadavg(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        try:
            with open("/proc/loadavg", "r", encoding="utf-8") as f:
                parts = f.read().strip().split()
            return float(parts[0]), float(parts[1]), float(parts[2])
        except Exception:
            return None, None, None

    def _read_rss_bytes_process(self) -> Optional[int]:
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as f:
                parts = f.read().strip().split()
            resident_pages = int(parts[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            return None

    def _read_cpu_freq_summary_mhz(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        cpu_freqs = []
        for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cpu_freqs.append(int(f.read().strip()) / 1000.0)  # kHz -> MHz
            except Exception:
                continue
        if not cpu_freqs:
            return None, None, None
        return min(cpu_freqs), sum(cpu_freqs) / len(cpu_freqs), max(cpu_freqs)

    def _emit_rapl_periodic_snapshot(self):
        if not self._rapl_domains:
            return

        phase_name = self.current_phase
        iteration = self.current_iteration
        cpu_util = self._read_cpu_util_pct_total()
        load1, load5, load15 = self._read_loadavg()
        rss_bytes = self._read_rss_bytes_process()
        freq_min, freq_mean, freq_max = self._read_cpu_freq_summary_mhz()

        for domain in self._rapl_domains:
            domain_path = Path(domain["domain_path"])
            error_fields = list(domain.get("init_error_fields", []))
            energy_uj = self._safe_call(
                "cpu_energy_uJ",
                lambda: self._read_int_file(domain_path / "energy_uj"),
                error_fields,
            )

            rec = self._base_record(
                phase_name=phase_name,
                record_type="PERIODIC",
                iter_id=iteration,
            )
            rec.update(
                {
                    "cpu_energy_uJ": energy_uj,
                    "max_energy_range_uJ": domain.get("max_energy_range_uJ"),
                    "rapl_domain": domain.get("rapl_domain"),
                    "domain_path": domain.get("domain_path"),
                    "cpu_util_pct_total": cpu_util,
                    "load1": load1,
                    "load5": load5,
                    "load15": load15,
                    "rss_bytes_process": rss_bytes,
                    "cpu_freq_min_MHz": freq_min,
                    "cpu_freq_mean_MHz": freq_mean,
                    "cpu_freq_max_MHz": freq_max,
                    "error_fields": error_fields,
                }
            )
            self._append_jsonl(self.rapl_periodic_file, rec)

    def mark_phase_start(self, phase_name: PhaseType, iteration: int = None):
        """Mark the start of a training phase."""
        if not self.enabled:
            return
        if self._phase_open:
            self._emit_nvml_boundary_snapshot(self.current_phase, "END", self.current_iteration)
            self._emit_rapl_boundary_snapshot(self.current_phase, "END", self.current_iteration)
            self._phase_open = False

        self.current_phase = phase_name
        if iteration is not None:
            self.current_iteration = iteration
        self.phase_start_time = time.time()
        self._phase_start_monotonic_ns[phase_name] = time.monotonic_ns()
        self._phase_open = True

        self._write_state(
            {
                "phase_id": PHASE_IDS[phase_name],
                "phase_name": phase_name,
                "iteration": self.current_iteration,
                "timestamp": self.phase_start_time,
            }
        )
        self._emit_nvml_boundary_snapshot(phase_name, "START", self.current_iteration)
        self._emit_rapl_boundary_snapshot(phase_name, "START", self.current_iteration)

    def mark_phase_end(self, phase_name: PhaseType = None):
        """Mark the end of a training phase."""
        if not self.enabled:
            return 0.0

        final_phase = phase_name or self.current_phase
        self._emit_nvml_boundary_snapshot(final_phase, "END", self.current_iteration)
        self._emit_rapl_boundary_snapshot(final_phase, "END", self.current_iteration)
        self._phase_open = False
        if self.phase_start_time:
            duration = time.time() - self.phase_start_time
            return duration
        return 0.0

    def log_tokens_and_steps(
        self,
        payload: Dict[str, Any],
        phase_name: Optional[str] = None,
        iteration: Optional[int] = None,
    ):
        """Log token/step denominators needed for J/token calculations."""
        if not self.enabled:
            return
        iter_id = self.current_iteration if iteration is None else iteration
        phase = self.current_phase if phase_name is None else phase_name
        record = self._base_record(
            phase_name=phase,
            record_type="PERIODIC",
            iter_id=iter_id,
        )
        record["metric_scope"] = "tokens_and_steps"
        record.update(payload)
        record.setdefault("error_fields", [])
        self._append_jsonl(self.tokens_file, record)

    def cleanup(self):
        """Clean up resources."""
        if self._cleanup_done:
            return
        self._cleanup_done = True
        if not self.enabled:
            return
        self._stop_event.set()
        if self._periodic_thread is not None:
            self._periodic_thread.join(timeout=2.0)
        if self._phase_open:
            try:
                self._emit_nvml_boundary_snapshot(self.current_phase, "END", self.current_iteration)
                self._emit_rapl_boundary_snapshot(self.current_phase, "END", self.current_iteration)
            except Exception as e:
                print(f"Warning: failed to emit final phase END snapshot during cleanup: {e}")
            self._phase_open = False
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        if self.state_file.exists():
            try:
                self.state_file.unlink()
            except Exception as e:
                print(f"Warning: Cleanup failed: {e}")


class PhaseReader:
    """Reader class - used by monitoring script to query current phase."""

    def __init__(self, experiment_name: str):
        scratch_dir = _require_env_str("SCRATCH_DIR")
        monitoring_dir = (
            os.environ.get("MONITORING_DIR")
            or os.environ.get("VERL_FILE_LOGGER_ROOT")
            or f"{scratch_dir}/logs"
        )
        self.state_file = Path(monitoring_dir) / f"phase_state_{experiment_name}.json"

    def get_current_phase(self) -> Dict:
        """Read the current phase state."""
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except Exception:
            return {
                "phase_id": 0,
                "phase_name": "idle",
                "iteration": 0,
                "timestamp": time.time(),
            }


class SubPhaseProfiler(PhaseProfiler):
    """
    Enhanced profiler that captures sub-phase timings in addition to phase transitions.

    Inherits all functionality from PhaseProfiler and adds timing log capability.
    Can operate in two modes:
    - granularity='phase': Only track phase-level (same as PhaseProfiler)
    - granularity='operation': Track operation-level timings (sub-phase)
    """

    def __init__(self, experiment_name: str, enable: bool = True, granularity: str = "phase"):
        """
        Initialize sub-phase profiler.

        Args:
            experiment_name: Unique name for this experiment
            enable: Whether profiling is enabled
            granularity: 'phase' for phase-level only, 'operation' for sub-phase tracking
        """
        # Initialize parent class (handles phase state file)
        super().__init__(experiment_name, enable)

        if not self.enabled:
            return

        self.granularity = granularity

        # Only create timing log if we're doing operation-level profiling
        if self.granularity == "operation":
            # Create timing log file (JSONL format - one JSON object per line)
            monitoring_dir = self.data_dir
            monitoring_dir.mkdir(parents=True, exist_ok=True)
            self.timing_log_file = monitoring_dir / f"phase_timings_{experiment_name}.jsonl"

            # Clear any existing log file
            if self.timing_log_file.exists():
                self.timing_log_file.unlink()

            print(f"✓ Sub-phase profiler initialized (granularity: {granularity})")
            print(f"  Phase state: {self.state_file}")
            print(f"  Timing log: {self.timing_log_file}")
        else:
            print(f"✓ Phase profiler initialized (granularity: {granularity})")
            print(f"  Phase state: {self.state_file}")

    def log_timings(self, timing_dict: Dict[str, float], phase_name: str, iteration: int):
        """
        Log timing data for sub-phase analysis.

        This captures the timing_raw dictionary from verl's marked_timer instrumentation
        and emits one JSONL record per subphase metric, each explicitly tagged with
        phase_name + subphase_name + iteration.

        Not every key in timing_raw is guaranteed to be a duration. Keys that look like
        ratios/imbalances are emitted as generic metrics (value + unit).

        Only logs if granularity is set to 'operation'.

        Args:
            timing_dict: Dictionary of operation names to numeric metrics (from marked_timer)
            phase_name: Current phase name (rollout, rl_policy, training, validation, other)
            iteration: Current training iteration
        """
        if not self.enabled:
            return

        # Only log timings if we're doing operation-level profiling
        if self.granularity != "operation":
            return

        for subphase_name, raw_value in timing_dict.items():
            error_fields: List[str] = []
            try:
                metric_value = float(raw_value)
            except Exception:
                metric_value = None
                error_fields.append("metric_value")

            name_lower = str(subphase_name).lower()
            metric_unit = "s"
            if any(token in name_lower for token in ("ratio", "imbalance", "percent", "pct", "fraction", "frac")):
                metric_unit = "ratio"

            timing_entry = self._base_record(
                phase_name=phase_name,
                record_type="PERIODIC",
                iter_id=iteration,
            )
            # Slim timing schema: drop constant/derivable duplicates.
            timing_entry.pop("timestamp", None)
            timing_entry.pop("record_type", None)
            timing_entry.update(
                {
                    "subphase_name": subphase_name,
                    "value": metric_value,
                    "metric_unit": metric_unit if metric_value is not None else None,
                    "error_fields": error_fields,
                }
            )
            self._append_jsonl(self.timing_log_file, timing_entry)

    def cleanup(self):
        """Clean up resources including timing log file."""
        super().cleanup()
        if self.enabled and hasattr(self, "timing_log_file") and self.timing_log_file.exists():
            try:
                # Don't delete timing log - it's valuable data!
                # self.timing_log_file.unlink()
                pass
            except Exception as e:
                print(f"Warning: Timing log cleanup issue: {e}")


__all__ = [
    "SubPhaseProfiler",
    "PhaseProfiler",
    "PhaseReader",
    "PHASE_IDS",
    "PhaseType",
]
