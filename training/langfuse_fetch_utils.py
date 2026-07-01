from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, TypeVar

T = TypeVar("T")


def _retry_call(fn: Callable[[], T], retries: int = 1) -> T:
    attempts = max(0, int(retries)) + 1
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            if attempt >= attempts - 1:
                raise
            time.sleep(min(2.0, 0.2 * (2 ** attempt)))
    if last_exc is not None:  # pragma: no cover
        raise last_exc
    raise RuntimeError("Unreachable retry state")


def fetch_run_traces(
    langfuse: Any,
    dataset_name: str,
    run_name: str,
    retries: int = 1,
) -> tuple[List[Any], Dict[str, Any]]:
    run = _retry_call(
        lambda: langfuse.api.datasets.get_run(
            dataset_name=dataset_name,
            run_name=run_name,
        ),
        retries=retries,
    )
    run_items = list(getattr(run, "dataset_run_items", []) or [])
    traces_by_id: Dict[str, Any] = {}
    for run_item in run_items:
        trace_id = str(getattr(run_item, "trace_id", None) or "")
        if not trace_id:
            continue
        try:
            trace = _retry_call(
                lambda tid=trace_id: langfuse.api.trace.get(trace_id=tid),
                retries=retries,
            )
        except Exception:
            continue
        traces_by_id[trace_id] = trace
    return run_items, traces_by_id
