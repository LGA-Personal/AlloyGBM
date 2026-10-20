"""A monotonic fit budget shared by automatic-round probe and refit phases."""

from contextlib import contextmanager
from contextvars import ContextVar
import math
import time

_deadline = ContextVar("alloygbm_fit_deadline", default=None)


def validate_time_limit(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("time_limit must be a finite positive number or None")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("time_limit must be finite and positive")
    return value


def remaining_seconds(limit, started):
    if limit is None:
        return None
    remaining = limit - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("fit time limit exhausted during input preparation")
    return remaining


@contextmanager
def fit_budget(time_limit):
    seconds = validate_time_limit(time_limit)
    parent = _deadline.get()
    end = None if seconds is None else time.monotonic() + seconds
    if parent is not None:
        end = parent if end is None else min(parent, end)
    token = _deadline.set(end)
    try:
        yield
    finally:
        _deadline.reset(token)


def native_budget_kwargs():
    end = _deadline.get()
    if end is None:
        return {}
    remaining = end - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("fit time limit exhausted during input preparation")
    return {"max_fit_seconds": remaining}
