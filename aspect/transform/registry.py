"""Transform function registry."""

from collections.abc import Callable


FUNCTION_REGISTRY: dict[str, Callable] = {}
COLLATOR_REGISTRY: dict[str, str] = {}


def register_function(
    name: str,
    *,
    collator: str | None = None,
):
    """Register a transform and optional runtime collator."""

    def decorator(fn: Callable):
        FUNCTION_REGISTRY[name] = fn
        if collator is not None:
            COLLATOR_REGISTRY[name] = collator
        return fn

    return decorator
