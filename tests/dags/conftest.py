"""Mock airflow.sdk before any DAG file is imported.

Airflow 3.x requires Python <3.14, so it cannot be installed in this dev
environment. These mocks let the DAG module be imported and tested for
structure and logic without a real Airflow install.

Key design: @task wraps the function in a TaskProxy that records the call
graph but does NOT execute the task body. Calling a TaskProxy returns a
sentinel (another proxy), matching how Airflow's TaskFlow API works at
DAG-build time vs run time.
"""
import sys
from types import ModuleType


class _TaskProxy:
    """Wraps a task function. Calling it returns a sentinel, not the result."""
    def __init__(self, fn):
        self._fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    def __call__(self, *args, **kwargs):
        # At DAG-build time, calling a task returns a placeholder (XComArg).
        # We return self so downstream >> operators are also no-ops.
        return self

    def __rshift__(self, other):
        return other

    def __rrshift__(self, _other):
        return self

    # Support list wiring: branch >> [retrain, skip]
    def __iter__(self):
        return iter([self])


class _FakeTask:
    """Minimal stand-in for the @task namespace."""

    def __call__(self, *deco_args, **deco_kwargs):
        if deco_args and callable(deco_args[0]):
            # @task  (no parentheses)
            return _TaskProxy(deco_args[0])
        # @task() or @task(retries=1)
        return lambda fn: _TaskProxy(fn)

    def branch(self, *deco_args, **deco_kwargs):
        if deco_args and callable(deco_args[0]):
            return _TaskProxy(deco_args[0])
        return lambda fn: _TaskProxy(fn)


def _dag_decorator(*args, **kwargs):
    """@dag — call the decorated function so the DAG graph is built, then return it."""
    if args and callable(args[0]):
        fn = args[0]
        fn()
        return fn

    def wrapper(fn):
        fn()
        return fn
    return wrapper


# Patch list wiring so `branch >> [retrain, skip]` doesn't raise
_orig_list_rrshift = list.__rrshift__ if hasattr(list, "__rrshift__") else None


# Build the fake airflow hierarchy
_airflow = ModuleType("airflow")
_airflow_sdk = ModuleType("airflow.sdk")
_airflow_sdk.dag = _dag_decorator
_airflow_sdk.task = _FakeTask()

sys.modules.setdefault("airflow", _airflow)
sys.modules.setdefault("airflow.sdk", _airflow_sdk)
