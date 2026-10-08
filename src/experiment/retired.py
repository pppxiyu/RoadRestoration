"""Keep historical helpers importable without running the retired problem."""
from functools import wraps


def retired_problem(function):
    """Reject old experiment/generation entry points before any file changes."""
    @wraps(function)
    def blocked(*args, **kwargs):
        raise RuntimeError(
            f"{function.__module__}.{function.__name__} belongs to the retired problem. "
            "Use main.py with the finalized daily setting. For historical reproduction, "
            "use the preserved source snapshot in legacy/ or the original run's config/, "
            "in a separate checkout; do not mix historical and current settings.")
    return blocked
