"""Custom types."""

from typing import TYPE_CHECKING, Any, TypeAlias
from collections.abc import Iterable, Mapping, Union

from datasets import Dataset, IterableDataset
from pandas import DataFrame
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .transform.base import ColumnTransform
else:
    ColumnTransform = Any

DataLike: TypeAlias = Union[
    str, 
    DataFrame, 
    Mapping[str, ArrayLike], 
    Dataset, 
    IterableDataset
]
FeatureLike: TypeAlias = Union[
    str, 
    Mapping[str, ColumnTransform], 
    Mapping[str, Mapping[str, Any]], 
    Iterable[Union[
        str, 
        Mapping[str, ColumnTransform], 
        Mapping[str, Mapping[str, Any]]
    ]]
]
StrOrIterableOfStr = Union[str, Iterable[str]]

Datum: TypeAlias = Mapping[str, Any]
Batch: TypeAlias = Mapping[str, Any]
