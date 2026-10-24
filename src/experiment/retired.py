"""Keep historical helpers importable without running the retired problem."""
from functools import wraps


def retired_problem(function):
    """Reject old experiment/generation entry points before any file changes."""
    @wraps(function)
    def blocked(*args, **kwargs):
        raise RuntimeError(
            f"{function.__module__}.{function.__name__} belongs to the retired problem. "
            "Use main.py with an explicit finalized daily branch. Historical source and "
            "results are retained for reference in .legacy/LATEST.md and the original "
            "run's config/. Do not start or resume historical experiments.")
    return blocked
