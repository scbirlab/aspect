"""Data preprocessing functions."""

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union
from functools import cache, partial
import hashlib

import numpy as np

from .registry import register_function


@register_function("identity")
def Identity() -> Callable:
    """Simple pass-through.
    
    """
    def _identity(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> np.ndarray:
        return np.asarray(data[input_column])
    return _identity


@register_function("log")
def Log() -> Callable:
    """Log10 of float.
    
    """
    def _log(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> np.ndarray:
        return np.log(np.asarray(data[input_column]))
    return _log


@register_function("one-hot")
def OneHot(
    categories: Iterable[Union[str, int]],
    intercept: bool = False
) -> Callable:
    """Convert string labels into one-hot encodings.
    
    """
    prepend = [1] if intercept else []

    def _one_hot(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> np.ndarray:
        return np.asarray([
            prepend + [1 if x == cat else 0 for cat in categories] 
            for x in data[input_column]
        ])

    return _one_hot


@register_function("hash")
def Hash(
    ndim: int = 256,
    hash_name: str = "sha1",
    dense: Optional[int] = None,
    normalize: bool = False,
    seed: int = 42,
) -> Callable:
    """
    Deterministic string hashing featurizer.

    Each input string is hashed to a fixed-length numeric vector in [0,1] or [-1,1].
    
    Parameters
    ==========
    ndim : int
        Output feature dimension (number of hash buckets).
    hash_name : str
        Hash function to use ('sha1', 'md5', 'blake2b', ...).
    dense : int, optional
        Dense projection to this number of dimensions.
    normalize : bool
        Whether to normalize vector to unit length (L2 norm = 1).
    seed : int
        Seed to vary hash folding (adds reproducible offset).

    Returns
    =======
    Callable
        Function mapping (data, input_column) → np.ndarray [N, n_features]

    Examples
    ========
    >>> import numpy as np
    >>> f = Hash(ndim=8, seed=0)
    >>> X = f({"assay": ["MIC", "MIC", "binding"]}, "assay")
    >>> X.shape
    (3, 8)
    >>> np.array_equal(X[0], X[1])
    True
    >>> np.array_equal(X[0], X[2])
    False

    Seed changes output:
    >>> f2 = Hash(ndim=8, seed=1)
    >>> X2 = f2({"assay": ["MIC"]}, "assay")
    >>> np.array_equal(X2[0], X[0])
    False

    Different hash backend changes output:
    >>> f_md5 = Hash(ndim=8, hash_name="md5", seed=0)
    >>> np.array_equal(f_md5({"assay": ["MIC"]}, "assay")[0], X[0])
    False

    Dense projection has the right shape and dtype:
    >>> g = Hash(ndim=8, dense=4, seed=0)
    >>> Y = g({"assay": ["MIC", "binding"]}, "assay")
    >>> Y.shape, Y.dtype == np.float32
    ((2, 4), True)

    Values are in [-1, 1] before normalization:
    >>> v = f({"assay": ["MIC"]}, "assay")[0]
    >>> (v.min() >= -1 - 1e-6 and v.max() <= 1 + 1e-6).item()
    True

    Non-string values are stringified:
    >>> f({"assay": [123, None]}, "assay").shape
    (2, 8)

    """
    from numpy.random import default_rng
    if ndim <= 0:
        raise ValueError("ndim must be > 0")
    if dense is not None and dense <= 0:
        raise ValueError("dense must be > 0 when provided")

    if dense is not None:
        generator = default_rng(seed=seed)
        projector = generator.normal(
            scale=1. / np.sqrt(dense), 
            size=(int(ndim), int(dense)),
        ).astype(np.float32)
    else:
        projector = None

    def _hash_single(s: str) -> np.ndarray:
        # Stable deterministic hashing per string
        if not isinstance(s, str):
            s = str(s)
        h = hashlib.new(hash_name)
        h.update((s + str(seed)).encode("utf-8"))
        digest = np.frombuffer(h.digest(), dtype=np.uint8)
        # Repeat or truncate digest to reach n_features bytes
        vec = np.resize(digest, ndim)
        vec = (2. * vec / 255. - 1.)  # map to [-1, 1]
        return vec
    

    def _hash(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> np.ndarray:
        vectors = np.stack([
            _hash_single(v) for v in data[input_column]
        ], axis=0).astype(np.float32)

        if projector is not None:
            vectors = vectors @ projector
        if normalize:
            n = np.linalg.norm(vectors, axis=-1, keepdim=True)
            nz = n > 0
            vectors[nz[:, 0]] = vectors[nz[:, 0]] / n[nz]
        return vectors

    return _hash
    

@register_function("morgan-fingerprint")
def MorganFingerprint(**kwargs) -> Callable:
    """Get Morgan fingerprint from SMILES.
    
    """
    try:
        from schemist.features import calculate_feature
    except ImportError:
        raise ImportError(f"schemist not installed! Try `pip install duvidnn[chem]`.")
    feature_calculator = partial(
        calculate_feature,
        feature_type="fp",
        return_dataframe=False,
        on_bits=False,
        **kwargs,
    )

    def _morgan_fingerprint(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> np.ndarray:
        fingerprints, _ = feature_calculator(strings=data[input_column])
        return fingerprints
        
    return _morgan_fingerprint


@register_function("descriptors-2d")
def Descriptors2D(
    normalized: bool = True,
    histogram_normalized: bool = True
) -> Callable:
    """Get 2D descriptors from SMILES, optionally normalized.
    
    """
    try:
        from schemist.features import calculate_feature
    except ImportError:
        raise ImportError(f"schemist not installed! Try `pip install duvidnn[chem]`.")

    feature_calculator = partial(
        calculate_feature,
        feature_type="2d",
        return_dataframe=False,
        normalized=normalized,
        histogram_normalized=histogram_normalized,
    )
    
    def _descriptors_2d(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> np.ndarray:
        desc_2d, _ = feature_calculator(strings=data[input_column])
        return desc_2d

    return _descriptors_2d


@register_function("vectome-fingerprint")
def VectomeFingerprint(
    method: str = "countsketch",
    ndim: int = 2048,
    check_spelling: bool = True,
    **kwargs
) -> Callable:
    """Get MinHash fingerprint from species name or taxon ID.
    
    """
    try:
        from vectome.vectorize import vectorize
    except ImportError:
        raise ImportError(f"Vectome not installed! Try `pip install duvidnn[bio]`.")

    feature_calculator = cache(partial(
        vectorize,
        method=method,
        dim=ndim,
        check_spelling=check_spelling,
        quiet=True,
        **kwargs,
    ))

    def _vectome_fingerprint(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> np.ndarray:
        return feature_calculator(query=tuple(data[input_column]))
        
    return _vectome_fingerprint


@register_function("chemprop-mol")
def ChempropData(
    label_column: Optional[Union[str, Iterable[str]]] = None,
    extra_featurizers: Optional[Union[Mapping[str, Any], Iterable[Mapping[str, Any]]]] = None
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
        raise ImportError("Chemprop not installed. Try `pip install duvidnn[chem]`.")

    if isinstance(extra_featurizers, str):
        extra_featurizers = [extra_featurizers]
    if isinstance(label_column, str):
        label_column = [label_column]

    def _stack_columns(
        data: Mapping[str, Iterable],
        nrows: int,
        columns: Optional[Iterable[str]] = None
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
    ) -> List[Dict[str, np.ndarray]]:
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
