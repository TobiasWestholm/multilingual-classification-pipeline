from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional


# Conservative defaults: keep integrity checks robust, but avoid long idle waits.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_DELETE_WAIT_TIMEOUT_SEC = 60.0
DEFAULT_DELETE_WAIT_POLL_SEC = 1.0
DEFAULT_VERIFY_TIMEOUT_SEC = 120.0
DEFAULT_VERIFY_POLL_SEC = 1.0
DEFAULT_VERIFY_STABLE_SEC = 20.0


@dataclass(frozen=True)
class LangfusePostTiming:
    max_attempts: int
    delete_wait_timeout_sec: float
    delete_wait_poll_sec: float
    verify_timeout_sec: float
    verify_poll_sec: float
    verify_stable_sec: float


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp_positive(value: float, minimum: float = 0.1) -> float:
    return max(float(minimum), float(value))


def resolve_langfuse_post_timing_from_config(config: Mapping[str, Any]) -> LangfusePostTiming:
    """Resolve Langfuse post timing from trainer config keys."""
    return LangfusePostTiming(
        max_attempts=max(1, _to_int(config.get("langfuse_post_max_attempts"), DEFAULT_MAX_ATTEMPTS)),
        delete_wait_timeout_sec=_clamp_positive(
            _to_float(config.get("langfuse_delete_wait_timeout_sec"), DEFAULT_DELETE_WAIT_TIMEOUT_SEC),
            minimum=1.0,
        ),
        delete_wait_poll_sec=_clamp_positive(
            _to_float(config.get("langfuse_delete_wait_poll_sec"), DEFAULT_DELETE_WAIT_POLL_SEC)
        ),
        verify_timeout_sec=_clamp_positive(
            _to_float(config.get("langfuse_post_verify_timeout_sec"), DEFAULT_VERIFY_TIMEOUT_SEC),
            minimum=1.0,
        ),
        verify_poll_sec=_clamp_positive(
            _to_float(config.get("langfuse_post_verify_poll_sec"), DEFAULT_VERIFY_POLL_SEC)
        ),
        verify_stable_sec=max(
            0.0,
            _to_float(config.get("langfuse_post_verify_stable_sec"), DEFAULT_VERIFY_STABLE_SEC),
        ),
    )


def resolve_langfuse_post_timing_from_env(env: Optional[Mapping[str, str]] = None) -> LangfusePostTiming:
    """Resolve Langfuse post timing from env vars used by human-eval posting."""
    source = env if env is not None else os.environ
    return LangfusePostTiming(
        max_attempts=max(1, _to_int(source.get("LANGFUSE_POST_MAX_ATTEMPTS"), DEFAULT_MAX_ATTEMPTS)),
        delete_wait_timeout_sec=_clamp_positive(
            _to_float(source.get("LANGFUSE_DELETE_WAIT_TIMEOUT_SEC"), DEFAULT_DELETE_WAIT_TIMEOUT_SEC),
            minimum=1.0,
        ),
        delete_wait_poll_sec=_clamp_positive(
            _to_float(source.get("LANGFUSE_DELETE_WAIT_POLL_SEC"), DEFAULT_DELETE_WAIT_POLL_SEC)
        ),
        verify_timeout_sec=_clamp_positive(
            _to_float(source.get("LANGFUSE_POST_VERIFY_TIMEOUT_SEC"), DEFAULT_VERIFY_TIMEOUT_SEC),
            minimum=1.0,
        ),
        verify_poll_sec=_clamp_positive(
            _to_float(source.get("LANGFUSE_POST_VERIFY_POLL_SEC"), DEFAULT_VERIFY_POLL_SEC)
        ),
        verify_stable_sec=max(
            0.0,
            _to_float(source.get("LANGFUSE_POST_VERIFY_STABLE_SEC"), DEFAULT_VERIFY_STABLE_SEC),
        ),
    )
