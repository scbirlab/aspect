"""Data pipeline class."""

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Tuple, Optional, Union

from functools import cached_property, partial
import os

from carabiner import cast, print_err

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset
else:
    Dataset, DatasetDict, IterableDataset = Any, Any, Any

import numpy as np
from numpy.typing import ArrayLike

from . import app_name, __version__
from .checkpoint_utils import load_checkpoint_file, save_json
from .io import AutoDataset
from .package_data import CACHE_DIR
from .transform.base import ColumnTransform
from .typing import DataLike, StrOrIterableOfStr

DEFAULT_BATCH_SIZE: int = 1024
DEFAULT_FORMAT: str = "numpy"

def _check_column_presence(
    features: StrOrIterableOfStr,
    data: Dataset
) -> Iterable[str]:
    columns = cast(features, to=list)
    data_cols = data.column_names
    absent_columns = [col for col in columns if col not in data_cols]
    if absent_columns:
        raise ValueError(
            f"""
            Requested columns ({', '.join(columns)}) not present in 
            {type(data)}: {', '.join(absent_columns)}.
            """
        )
    return columns


def _check_is_calculated(
    x: Dataset,  
    column_transform: ColumnTransform
) -> Tuple[str, bool]:
    """Check named column is in dataset.

    Examples
    ========
    _check_column_presence passes when columns present:

    >>> from unittest.mock import MagicMock
    >>> ds = MagicMock()
    >>> ds.column_names = ["smiles", "mic"]
    >>> _check_column_presence(["smiles", "mic"], ds)
    ['smiles', 'mic']

    _check_column_presence raises on absent columns:

    >>> _check_column_presence(["smiles", "missing"], ds)
    Traceback (most recent call last):
        ...
    ValueError: ...missing...

    """
    out_column = column_transform.output_column
    return out_column, out_column in x.column_names


def _fill_na(
    x: Mapping[str, Any],
    types: Mapping[str, Any]
) -> Dict[str, Any]:
    """Fill missing values with typed missing.

    For example, numeric filled with zeros, and strings filled with `""`.
    Examples
    ========
    _fill_na fills by dtype:

    >>> _fill_na(
    ...     {"a": [1, None, 3], "b": ["x", None, "z"], "c": [1.0, None, 3.0]},
    ...     {"a": "int64", "b": "string", "c": "float32"}
    ... )
    {'a': [1, 0, 3], 'b': ['x', '', 'z'], 'c': [1.0, 0.0, 3.0]}

    _fill_na unknown dtype fills None:

    >>> _fill_na({"a": [1, None]}, {"a": "bool"})
    {'a': [1, None]}

    """
    for key in x:
        this_type = types[key]
        if this_type.startswith(("int", "uint")):
            fill_value = 0
        elif this_type.startswith("float"):
            fill_value = 0.
        elif this_type in ("string", "large_string"):
            fill_value = ""
        else:
            fill_value = None
        
        x[key] = [fill_value if v is None else v for v in x[key]]
    return x


class DataPipeline:
    """Data processing pipeline.

    Examples
    ========
    Construction from dict spec:

    >>> p = DataPipeline({"log_affinity": ("affinity", "log")})
    >>> "log_affinity" in p.column_transforms
    True
    >>> len(p.column_transforms["log_affinity"])
    1
    >>> p.column_transforms["log_affinity"][0].input_column
    'affinity'

    Construction from list of 2-tuples:

    >>> p2 = DataPipeline([("affinity", "log"), ("assay", ["hash"])])
    >>> sorted(p2.column_transforms.keys())
    ['col_00', 'col_01']

    Chained transforms propagate input → output columns:

    >>> p3 = DataPipeline({"feat": ("assay", ["hash", "identity"])})
    >>> chain = p3.column_transforms["feat"]
    >>> chain[1].input_column == chain[0].output_column
    True

    serialize_transforms produces JSON-compatible dicts:

    >>> serialized = p.serialize_transforms(p.column_transforms)
    >>> isinstance(serialized["log_affinity"], tuple)
    True
    >>> isinstance(serialized["log_affinity"][0], dict)
    True
    >>> serialized["log_affinity"][0]["name"]
    'log'

    Invalid column spec raises:

    >>> DataPipeline({"bad": ("affinity", "log", "extra")})
    Traceback (most recent call last):
        ...
    ValueError: Column transforms must be 2-tuples...

    Non-string first element raises:

    >>> DataPipeline({"bad": (123, "log")})
    Traceback (most recent call last):
        ...
    ValueError: First item must be input_column name, or dict with input_column key.

    __call__ produces output columns:

    >>> import numpy as np
    >>> p = DataPipeline({"log_affinity": ("affinity", "log")})
    >>> data = {"affinity": [1.0, 10.0, 100.0]}
    >>> out = p(data)
    >>> "log_affinity" in out.column_names
    True
    >>> np.allclose(out["log_affinity"], np.log([1., 10., 100.])[:,None])
    True

    drop_unused_columns removes input columns:

    >>> out2 = p(data, drop_unused_columns=True)
    >>> "affinity" not in out2.column_names
    True
    >>> "log_affinity" in out2.column_names
    True

    keep_extra_columns preserved when dropping:

    >>> data2 = {"affinity": [1.0, 10.0], "label": [0, 1]}
    >>> out3 = p(data2, drop_unused_columns=True, keep_extra_columns=["label"])
    >>> "label" in out3.column_names
    True

    Missing input column raises:

    >>> p({"wrong_col": [1.0, 2.0]})
    Traceback (most recent call last):
        ...
    ValueError: ...affinity...

    """
    def __init__(
        self,
        column_transforms: Optional[Iterable[Union[str, ColumnTransform]]] = None,
        columns_to_keep: Optional[Iterable[Union[str, ColumnTransform]]] = None,
        output_format: str = DEFAULT_FORMAT,
        output_format_opts: Optional[Mapping[str, Any]] = None,
        cache_dir: Optional[str] = None,
        _version: str = __version__,
        _app: str = app_name
    ):
        self._column_transforms = column_transforms or []
        columns_to_keep = columns_to_keep or []
        if isinstance(columns_to_keep, str):
            columns_to_keep = [columns_to_keep]
        self.columns_to_keep = columns_to_keep
        self._version = _version
        self._app = _app
        self.column_transforms = self.canonicalize_transforms(self._column_transforms)
        self.output_format = output_format
        self.output_format_opts = output_format_opts or {}
        self.data_in = None
        self.data_out = None
        self.data_out_example = None
        self.data_out_shape = None
        self.cache_dir = cache_dir or CACHE_DIR
        self._data_in_filename = "data-in.hf"
        self._data_out_filename = "data-out.hf"
        self._data_out_example_filename = "data-out-example.hf"
        self._data_loaded = False
        self._config_filename = "config.json"

        self._inspect_data_out()

    def __eq__(self, other):
        if hasattr(other, "column_transforms_serialized"):
            return all([
                self.column_transforms_serialized == other.column_transforms_serialized,
                self.columns_to_keep == other.columns_to_keep,
            ])
        else:
            raise ValueError(f"Cannot compare {type(self)} with {type(other)}.")

    @cached_property
    def column_transforms_serialized(self):
        return self.serialize_transforms(self.column_transforms)

    def _canonicalize_transforms(
        self,
        column_transforms: Iterable[Union[str, Mapping, ColumnTransform]],
        input_column: Optional[str] = None
    ) -> Tuple[ColumnTransform]:
        if isinstance(column_transforms, (str, dict, ColumnTransform)):
            column_transforms = [column_transforms]
        out = []
        prev_transform = None
        for i, candidate in enumerate(column_transforms):
            if prev_transform is not None:
                input_column = prev_transform.output_column
            elif input_column is None:
                if isinstance(candidate, ColumnTransform):
                    input_column = candidate.input_column
                elif isinstance(candidate, dict):
                    input_column = candidate["input_column"]
                else:
                    raise ValueError(
                        "Supply an input_column or a first ColumnTransform object."
                    )
            elif not isinstance(input_column, str):
                raise ValueError(
                        f"Supplied an input_column must be a str, but was {type(input_column)}: {input_column}."
                    )
            if isinstance(candidate, ColumnTransform):
                kwargs = candidate.to_dict()
            elif isinstance(candidate, dict):
                kwargs = candidate
            elif isinstance(candidate, str):
                kwargs = {"name": candidate}
            else:
                raise ValueError(
                    "Transform must be a ColumnTransform, dict, or str. "
                    f"It was {type(candidate)}: {candidate}"
                )
            transform = ColumnTransform(**({
                "_version": self._version, 
                "_app": self._app,
            } | kwargs | {
                "input_column": input_column,
            }))
            out.append(transform)
            prev_transform = transform
        return tuple(out)

    def canonicalize_transforms(
        self,
        column_transforms: Union[Mapping, Iterable],
        input_column: Optional[str] = None
    ) -> Dict[str, ColumnTransform]:
        if isinstance(column_transforms, (list, tuple)):
            if len(column_transforms) == 0:
                return {}
            elif isinstance(column_transforms[0], str):
                if isinstance(column_transforms[1], (list, tuple, dict, str, ColumnTransform)):
                    column_transforms = [[column_transforms]]
                else:
                    raise ValueError(
                        "Column transforms should be a list or dict of (input, [transforms...])"
                    )
        if not isinstance(column_transforms, dict):
            column_transforms = {
                f"col_{i:02d}": v 
                for i, v in enumerate(column_transforms)
            }
        for key in column_transforms:
            first_item = column_transforms[key][0]
            if isinstance(first_item, dict) and "input_column" in first_item:
                column_transforms[key] = (first_item["input_column"],  tuple(column_transforms[key]))
            elif isinstance(first_item, str):
                pass
            else:
                raise ValueError(f"First item must be input_column name, or dict with input_column key.")
        wrong_lengths = {k: len(v) for k, v in column_transforms.items() if len(v) != 2}
        if wrong_lengths:
            raise ValueError(
                "Column transforms must be 2-tuples. "
                f"These were not: {wrong_lengths}; {column_transforms}"
            )
        no_names = {k: v for k, v in column_transforms.items() if not isinstance(v[0], str)}
        if no_names:
            raise ValueError(
                "First item of column transform tuple must be string column name. "
                f"These were not: {no_names}"
            )
        out = {}
        for name, (input_column, subpipeline) in column_transforms.items():
            out[name] = self._canonicalize_transforms(
                subpipeline, 
                input_column=input_column,
            )
        return out

    def serialize_transforms(
        self, 
        column_transforms: Mapping[str, ColumnTransform]
    ) -> Dict[str, Tuple[dict]]:
            return {k: tuple(t.to_dict() for t in v) for k, v in column_transforms.items()}

    def _inspect_data_out(self) -> None:
        if self.data_out is not None:
            self.data_out = self.data_out.with_format(
                self.output_format, 
                **self.output_format_opts,
            )
            self.data_out_example = (
                self.data_out
                .take(1)
            )
            first_item = self.data_out_example.with_format("numpy")[:1]
            self.data_out_shape = {
                col: first_item[col].shape[1:]
                if not isinstance(first_item[col], dict)
                else {
                    k: v.shape[1:] if v is not None else None 
                    for k, v in first_item[col].items()
                }
                for col in self.data_out_example.column_names
            }
        return None

    def _resolve_data(
        self,
        data: DataLike, 
        cache_dir: Optional[str] = None
    ) -> Union[Dataset, IterableDataset]:
        return AutoDataset.load(data, cache=cache_dir or self.cache_dir)._dataset

    @staticmethod
    def _featurize(
        x: Mapping[str, ArrayLike],
        column_transforms: Mapping[str, dict]
    ) -> Dict[str, np.ndarray]:

        column_transforms = {
            k: [ColumnTransform(**d) for d in v]
            for k, v in column_transforms.items()
        }
        for name, transforms in column_transforms.items():
            if name in x:
                raise ValueError(
                    f"Output column name {name} already in data: {','.join(x)}. "
                    "Change transform name to avoid overwriting."
                )
        for name, transforms in column_transforms.items():
            prev_transform = None
            for i, transform in enumerate(transforms):
                if i == 0:
                    input_column = transform.input_column
                else:
                    input_column = prev_transform.output_column
                x = transform(x)
                prev_transform = transform
            x[name] = x[transform.output_column]
        return x

    @staticmethod
    def _unsqueeze(
        x: Mapping[str, ArrayLike]
    ) -> Dict[str, np.ndarray]:
        for key in x:
            vals = np.asarray(x[key])
            if vals.ndim == 1:
                x[key] = vals[:, None]
        return x

    def __call__(
        self, 
        dataset: DataLike, 
        batch_size: int = DEFAULT_BATCH_SIZE,
        drop_unused_columns: bool = False,
        keep_extra_columns: Optional[Iterable[str]] = None
    ):
        data_in = self._resolve_data(dataset)
        input_columns = sorted(set(
            seq[0].input_column for k, seq in self.column_transforms.items()
        ))  # get only the input column for each branch
        output_columns = sorted(set(self.column_transforms))

        if len(input_columns) == 0:
            raise AttributeError("No input columns specified.")
        
        _check_column_presence(
            input_columns, 
            data_in,
        )
        if drop_unused_columns:
            if keep_extra_columns is None:
                extra_cols = []
            else:
                extra_cols = list(keep_extra_columns)
            extra_cols = list(set(extra_cols + self.columns_to_keep).intersection(data_in.column_names))
            all_input_columns = list(set(input_columns + extra_cols))
            data_in = (
                data_in
                .select_columns(all_input_columns)
            )
            all_output_columns = output_columns + extra_cols
        else:
            all_input_columns = list(data_in.column_names)
            all_output_columns = all_input_columns + output_columns

        data_out = (
            data_in
            .map(
                _fill_na,
                fn_kwargs={
                    "types": {
                        key: f.dtype 
                        for key, f in data_in.info.features.items()
                    },
                },
                batched=True,
                batch_size=batch_size,
                desc="Filling NaN values",
            )
            .map(
                self._featurize,
                fn_kwargs={
                    "column_transforms": self.column_transforms_serialized,
                },
                batched=True,
                batch_size=batch_size,
                desc="Featurizing",
            )
        )
        data_out = (
            data_out
            .with_format(None)  # guard against tensors
            .select_columns(all_output_columns)
            .map(
                self._unsqueeze,
                batched=True,
                batch_size=batch_size,
                desc="Unsqueezing",
            )
        )
        self.data_in = data_in
        self.data_out = data_out
        self._data_loaded = True
        self._inspect_data_out()
        return self.data_out

    def save_checkpoint(
        self, 
        checkpoint_dir: str,
        skip_data_in: bool = False,
        skip_data_out: bool = False
    ):
        keys = {
            "kwargs": (
                ("column_transforms_serialized", "column_transforms"),
                "columns_to_keep",
                "output_format",
                "output_format_opts",
            ),
            "state": (
                "_column_transforms",
                "data_out_shape",
                "_data_in_filename",
                "_data_out_filename",
                "_data_out_example_filename",
                "_data_loaded",
                "_config_filename",
                "_version",
                "_app",
            ),
        }
        data_config = {
            section: {
                (key if isinstance(key, str) else key[1]): getattr(
                    self, 
                    (key if isinstance(key, str) else key[0]),
                ) 
                for key in section_keys
            }
            for section, section_keys in keys.items()
        }
        os.makedirs(checkpoint_dir, exist_ok=True)
        if self.data_in is not None and not skip_data_in:
            self.data_in.save_to_disk(
                os.path.join(checkpoint_dir, self._data_in_filename),
            )
        if self.data_out is not None and not skip_data_out:
            (
                self.data_out
                .save_to_disk(os.path.join(checkpoint_dir, self._data_out_filename)),
            )
            if self.data_out_example is not None:
                (
                    self.data_out_example
                    .save_to_disk(os.path.join(checkpoint_dir, self._data_out_example_filename)),
                )
        save_json(data_config, os.path.join(checkpoint_dir, self._config_filename))
        return None

    def load_checkpoint(
        self, 
        checkpoint: str,
        skip_data_in: bool = False,
        skip_data_out: bool = False,
        cache_dir: Optional[str] = None
    ):
        cache_dir = cache_dir or self.cache_dir
        data_config = load_checkpoint_file(
            checkpoint, 
            filename=self._config_filename,
            callback="json",
            none_on_error=False,
            cache_dir=cache_dir,
        )
        self.__init__(**data_config["kwargs"], cache_dir=cache_dir)
        for key, val in data_config["state"].items():
            setattr(self, key, val)
        if self._data_loaded:
            if not skip_data_in and self._data_loaded:
                self.data_in = load_checkpoint_file(
                    checkpoint, 
                    filename=self._data_in_filename,
                    callback="hf-dataset",
                    none_on_error=True,
                    cache_dir=cache_dir,
                )
            if not skip_data_out:
                self.data_out = load_checkpoint_file(
                    checkpoint, 
                    filename=self._data_out_filename,
                    callback="hf-dataset",
                    none_on_error=True,
                    cache_dir=cache_dir,
                )
            self._inspect_data_out()
        return self
