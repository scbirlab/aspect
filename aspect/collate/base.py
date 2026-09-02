"""Specialised column collation."""

from collections.abc import Callable, Mapping, Iterable
from importlib import import_module
from typing import Any, TypeAlias

from ..typing import Batch


CollateFn: TypeAlias = Callable[[Iterable[Any]], Any]


def resolve_collator(path: str) -> CollateFn:
    """Resolve ``module:attribute`` to a runtime collator."""

    try:
        module_name, attribute = path.split(":", 1)
    except ValueError as error:
        raise ValueError(
            "Collator path must have form "
            f"'module:attribute', but was {path!r}."
        ) from error

    module = import_module(module_name)
    collator = getattr(module, attribute)

    if not callable(collator):
        raise TypeError(
            f"Resolved collator {path!r} is not callable."
        )

    return collator


class ColumnCollator:
    """Override collation for selected dataset columns."""

    def __init__(
        self,
        collators: Mapping[str, CollateFn] | None = None,
    ):
        collators = collators or {}
        if not isinstance(collators, Mapping):
            raise ValueError(
                "If provided, `collators` must be a dict, "
                f"but was {type(collators)}: {collators}"
            )
        self.collators = dict(collators)

    def __call__(
        self,
        rows: Batch,
    ) -> dict[str, ...]:
        from torch.utils.data import default_collate

        out = {}
        if not rows:
            return out

        for column in rows[0]:
            values = [
                row[column]
                for row in rows
            ]

            if column in self.collators:
                out[column] = self.collators[column](values)
            else:
                out[column] = default_collate(values)

        return out
