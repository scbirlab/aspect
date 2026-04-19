"""Custom types."""

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Union

from datasets import Dataset, IterableDataset
from pandas import DataFrame
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .preprocessing.serializing import Preprocessor
else:
    Preprocessor = Any

DataLike = Union[
    str, 
    DataFrame, 
    Mapping[str, ArrayLike], 
    Dataset, 
    IterableDataset
]
FeatureLike = Union[
    str, 
    Mapping[str, Preprocessor], 
    Mapping[str, Mapping[str, Any]], 
    Iterable[Union[
        str, 
        Mapping[str, Preprocessor], 
        Mapping[str, Mapping[str, Any]]
    ]]
]
StrOrIterableOfStr = Union[str, Iterable[str]]
