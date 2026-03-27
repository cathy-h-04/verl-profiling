import time
import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch.distributed as dist

_PATCHED = False
_PATCH_LOCK = threading.Lock()
_COMM_STATE = threading.local()
_COMM_STATS: dict[str, float] = defaultdict(float)


def _record(op_name: str, duration_s: float) -> None:
    _COMM_STATS[op_name] += duration_s
    _COMM_STATS["total"] += duration_s


def _current_op() -> Optional[str]:
    return getattr(_COMM_STATE, "op_name", None)


def reset_comm_stats() -> None:
    _COMM_STATS.clear()


def get_comm_timing_stats() -> dict[str, float]:
    return dict(_COMM_STATS)


@contextmanager
def comm_context(op_name: str, reset: bool = False):
    prev = _current_op()
    _COMM_STATE.op_name = op_name
    if reset:
        reset_comm_stats()
    try:
        yield
    finally:
        _COMM_STATE.op_name = prev


class _WorkWrapper:
    def __init__(self, work: Any, op_name: str):
        self._work = work
        self._op_name = op_name

    def wait(self, *args: Any, **kwargs: Any):
        start = time.perf_counter()
        out = self._work.wait(*args, **kwargs)
        _record(self._op_name, time.perf_counter() - start)
        return out

    def __getattr__(self, name: str):
        return getattr(self._work, name)


def _wrap_collective(name: str, fn: Callable):
    def wrapper(*args: Any, **kwargs: Any):
        if not dist.is_initialized():
            return fn(*args, **kwargs)

        op_name = _current_op()
        if not op_name:
            return fn(*args, **kwargs)

        async_op = bool(kwargs.get("async_op", False))
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if async_op and hasattr(result, "wait"):
            return _WorkWrapper(result, op_name)
        _record(op_name, time.perf_counter() - start)
        return result

    wrapper.__name__ = getattr(fn, "__name__", name)
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    return wrapper


def enable_comm_timing() -> None:
    global _PATCHED
    if _PATCHED:
        return
    with _PATCH_LOCK:
        if _PATCHED:
            return
        _PATCHED = True

        collective_fns = [
            "all_reduce",
            "all_gather",
            "all_gather_into_tensor",
            "all_gather_object",
            "reduce_scatter",
            "reduce_scatter_tensor",
            "broadcast",
            "broadcast_object_list",
            "barrier",
            "monitored_barrier",
            "all_to_all",
            "all_to_all_single",
            "reduce",
        ]

        for fn_name in collective_fns:
            fn = getattr(dist, fn_name, None)
            if fn is None:
                continue
            setattr(dist, fn_name, _wrap_collective(fn_name, fn))
