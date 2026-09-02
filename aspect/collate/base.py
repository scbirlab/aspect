"""Specialised column collation."""

from collections.abc import Callable, Mapping, Iterable
from typing import Any, TypeAlias

from ..typing import Batch


CollateFn: TypeAlias = Callable[[Iterable[Any]], Any]


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
