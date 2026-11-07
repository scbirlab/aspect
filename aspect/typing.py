"""Custom types."""

from typing import TYPE_CHECKING, Any, TypeAlias, Union
from collections.abc import Iterable, Mapping

from datasets import Dataset, IterableDataset
from pandas import DataFrame
from providence.typing import DataLike
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .transform.base import ColumnTransform
else:
    ColumnTransform = Any

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
Batch: TypeAlias = Iterable[Mapping[str, Any]]
