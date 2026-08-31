"""Chemprop collation helpers."""

from collections.abc import Mapping, Iterable

from chemprop.data import MolGraph
from chemprop.data.collate import BatchMolGraph
import numpy as np
import torch
from torch import Tensor


def _as_tensor(x) -> Tensor:
    """Convert an value to a tensor."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x)


def _as_numpy(x) -> np.ndarray:
    """Convert an value to a tensor."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _molgraph_from_mapping(
    x: Mapping[str, ...]
) -> MolGraph:
    """Reconstruct a Chemprop MolGraph from a cached mapping."""
    return MolGraph(
        V=_as_numpy(x["V"]),
        E=_as_numpy(x["E"]),
        edge_index=_as_numpy(x["edge_index"]).astype(np.int32),
        rev_edge_index=_as_numpy(x["rev_edge_index"]).astype(np.int32),
    )


def _collate_optional(
    values: Iterable[...],
    *,
    stack: bool,
):
    """Collate an optional Chemprop feature."""
    if all(value is None for value in values):
        return None

    if any(value is None for value in values):
        raise ValueError(
            "Cannot collate Chemprop feature containing a mixture "
            "of None and non-None values."
        )

    tensors = [
        _as_tensor(value)
        for value in values
    ]

    if stack:
        return torch.stack(tensors, dim=0)

    return torch.cat(tensors, dim=0)


def chemprop_collate(
    values: Iterable[Mapping[str, ...]],
) -> dict[str, ...]:
    """Collate cached Chemprop features into a runtime model input.

    Each input value is expected to have the form::

        {
            "bmg": {
                "V": ...,
                "E": ...,
                "edge_index": ...,
                "rev_edge_index": ...,
            },
            "V_d": ...,
            "X_d": ...,
        }

    Returns
    =======
    dict
        Runtime representation suitable for ``ChempropEncoder(**batch)``.
    
    """
    if len(values) == 0:
        raise ValueError("Cannot collate an empty Chemprop batch.")

    molgraphs = [
        _molgraph_from_mapping(value["bmg"])
        for value in values
    ]
    bmg = BatchMolGraph(molgraphs)

    V_d = _collate_optional(
        [value.get("V_d") for value in values],
        stack=False,
    )
    X_d = _collate_optional(
        [value.get("X_d") for value in values],
        stack=True,
    )
    return {
        "bmg": bmg,
        "V_d": V_d,
        "X_d": X_d,
    }
