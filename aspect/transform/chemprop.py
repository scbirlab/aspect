"""Data preprocessing functions."""

from typing import Any
from collections.abc import Callable, Iterable, Mapping

import numpy as np

from .registry import register_function

@register_function(
    "chemprop-mol",
    collator="aspect.collate.chemprop:chemprop_collate",
)
def ChempropData(
    label_column: str | Iterable[str] | None = None,
    extra_featurizers: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None
) -> Callable:
    """Convert SMILES to iterable of Chemprop datum.
    
    """
    try:
        from chemprop.data import (
            MoleculeDatapoint, 
            MoleculeDataset, 
            MolGraph
        )
    except ImportError:
        raise ImportError("Chemprop not installed. Try `pip install aspect[chemprop]`.")

    if isinstance(extra_featurizers, str):
        extra_featurizers = [extra_featurizers]
    if isinstance(label_column, str):
        label_column = [label_column]

    def _stack_columns(
        data: Mapping[str, Iterable],
        nrows: int,
        columns: Iterable[str] | None = None
    ) -> np.ndarray:
        if columns is None:
            array = [None] * nrows
        else: 
            array = [np.asarray(data[col]) for col in columns]
            array = [a if a.ndim > 1 else a[..., np.newaxis] for a in array]
            if len(array) > 0:
                array = np.concatenate(array, axis=-1).astype(np.float32) 
            else:
                array = [None] * nrows
        return array
    
    def _chemprop_data(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> list[dict[str, np.ndarray]]:
        nrows = len(data[input_column])
        y_vals = _stack_columns(data, nrows, label_column)
        extra_features = _stack_columns(data, nrows, extra_featurizers)

        mol_datapoints = [
            MoleculeDatapoint.from_smi(smi=x, y=y, x_d=xd) 
            for x, y, xd in zip(data[input_column], y_vals, extra_features)
        ]
    
        datums = []
        for datum in MoleculeDataset(mol_datapoints):
            new_datum = {}
            for key, val in datum._asdict().items():
                if isinstance(val, MolGraph):
                    new_val = {
                        key2: val2.astype(np.float32) if isinstance(val2, np.ndarray) else np.float32(val2) 
                        for key2, val2 in val._asdict().items()
                    }
                elif isinstance(val, float):
                    new_val = np.float32(val)
                elif isinstance(val, np.ndarray):
                    new_val = val.astype(np.float32)
                elif val is not None:
                    new_val = val
                else:
                    new_val = None
                new_datum[key] = new_val
            datums.append(new_datum)
        return datums

    return _chemprop_data
