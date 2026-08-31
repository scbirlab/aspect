"""Specialised column collation."""

from collections.abc import Callable, Mapping, Iterable
from typing import Any, TypeAlias


CollateFn: TypeAlias = Callable[[Iterable[Any]], Any]


class ColumnCollator:
    """Override collation for selected dataset columns."""

    def __init__(
        self,
        collators: Mapping[str, CollateFn] | None = None,
    ):
        self.collators = collators or {}

    def __call__(
        self,
        rows: Iterable[Mapping[str, ...]],
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
