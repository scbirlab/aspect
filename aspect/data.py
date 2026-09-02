"""Data pipeline class."""

from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Iterable, Mapping
from functools import cached_property, partial
import json
import os
from pathlib import Path

from carabiner import cast, print_err

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset
    from .collate import ColumnCollator
else:
    Dataset, DatasetDict, IterableDataset = Any, Any, Any
    ColumnCollator = Any

import numpy as np
from numpy.typing import ArrayLike

from . import app_name, __version__
from .io import AutoDataset, DataSource, autoload, save_json, load_json
from .package_data import CACHE_DIR
from .transform.base import ColumnTransform
from .typing import DataLike, StrOrIterableOfStr


DEFAULT_BATCH_SIZE: int = 1024
DEFAULT_FORMAT: str = "numpy"
CONFIG_FILENAME = "config.json"
DATA_FILENAME = "data.parquet"
TRANSFORMED_FILENAME = "transformed.parquet"
EXAMPLE_FILENAME = "example.parquet"


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
) -> tuple[str, bool]:
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
) -> dict[str, Any]:
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
        column_transforms: Iterable[str | ColumnTransform] | None = None,
        columns_to_keep: Iterable[str | ColumnTransform] | None = None,
        output_format: str = DEFAULT_FORMAT,
        output_format_opts: Mapping[str, Any] | None = None,
        cache_dir: str | None = None,
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
        self.data_source = None
        self.data_out = None
        self.data_out_example = None
        self.data_out_shape = None
        self.cache_dir = cache_dir or CACHE_DIR
        self._data_loaded = False

        if self.data_out is not None or self.data_out_example is not None:
            self._inspect_data_out()

    def __eq__(self, other) -> bool:
        if hasattr(other, "column_transforms_serialized"):
            return all([
                self.column_transforms_serialized == other.column_transforms_serialized,
                self.columns_to_keep == other.columns_to_keep,
            ])
        else:
            raise ValueError(f"Cannot compare {type(self)} with {type(other)}.")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | str,
        *,
        cache_dir: str | None = None
    ) -> "DataPipeline":
        """Construct a DataPipeline from config or a JSON filename."""
        if isinstance(config, str):
            if not os.path.exists(config):
                raise FileNotFoundError(
                    "`config` was a string, but no file "
                    f"called `{config}` was found."
                )

            config = load_json(config)
        config = dict(config)
        # A saved checkpoint config contains pipeline and source sections.
        if "pipeline" in config:
            config = config["pipeline"]
        
        config_keys = {
            "column_transforms",
            "columns_to_keep",
            "output_format",
            "output_format_opts",
        }
        if set(config) <= config_keys:
            return cls(
                column_transforms=config.get(
                    "column_transforms",
                ),
                columns_to_keep=config.get(
                    "columns_to_keep",
                ),
                output_format=config.get(
                    "output_format",
                    DEFAULT_FORMAT,
                ),
                output_format_opts=config.get(
                    "output_format_opts",
                ),
                cache_dir=cache_dir,
            )
        return cls(
            column_transforms=config, 
            cache_dir=cache_dir,
        )

    def to_config(
        self,
        filename: str | os.PathLike | None = None,
    ) -> dict[str, Any]:
        """Return JSON-compatible constructor configuration."""
        config = {
            "column_transforms": self.column_transforms_serialized,
            "columns_to_keep": self.columns_to_keep,
            "output_format": self.output_format,
            "output_format_opts": self.output_format_opts,
        }
        # Round-trip through JSON
        config = json.loads(json.dumps(config))
        if filename is not None:
            save_json(config, str(filename))
        return config

    @cached_property
    def column_transforms_serialized(self):
        return self.serialize_transforms(self.column_transforms)

    def _canonicalize_transforms(
        self,
        column_transforms: Iterable[str | Mapping | ColumnTransform],
        input_column: str | None = None
    ) -> tuple[ColumnTransform]:
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
        column_transforms: Mapping[str, Any] | Iterable,
        input_column: str | None = None
    ) -> dict[str, ColumnTransform]:
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
    ) -> dict[str, tuple[dict]]:
            return {k: tuple(t.to_dict() for t in v) for k, v in column_transforms.items()}

    def _inspect_data_out(self) -> None:
        if self.data_out_example is None:
            if self.data_out is not None:
                self.data_out = self.data_out.with_format(
                    self.output_format, 
                    **self.output_format_opts,
                )
                self.data_out_example = (
                    self.data_out
                    .take(1)
                )
            else:
                raise AttributeError("Cannot inspect data without loading data or having an example")
        
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
        cache_dir: str | None = None
    ) -> Dataset | IterableDataset:
        resolved = AutoDataset.load(
            data, 
            cache=cache_dir or self.cache_dir,
        )
        self.data_source = resolved.source
        return resolved._dataset

    @staticmethod
    def _featurize(
        x: Mapping[str, ArrayLike],
        column_transforms: Mapping[str, dict]
    ) -> dict[str, np.ndarray]:

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
        x: Mapping[str, ArrayLike],
        columns: Iterable[str] | None = None
    ) -> dict[str, np.ndarray]:
        columns = columns or x.keys()
        for key in columns:
            vals = x[key]
            if not isinstance(vals, dict):
                vals = np.asarray(x[key])
                if vals.ndim == 1 and np.issubdtype(vals.dtype, np.number):
                    x[key] = vals[:, None]
        return x

    def __call__(
        self, 
        dataset: DataLike, 
        batch_size: int = DEFAULT_BATCH_SIZE,
        drop_unused_columns: bool = False,
        keep_extra_columns: Iterable[str] | None = None
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
                        key: f.dtype if hasattr(f, "dtype") 
                        else f.feature.dtype
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
                fn_kwargs={"columns": output_columns},
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

    def save(
        self,
        path: str | os.PathLike,
        *,
        save_transformed_columns: Iterable[str] | bool | None = None,
        save_source_data: bool | None = None,
        discard_example_data: bool = False
    ) -> None:
        """Save the pipeline and required training-data artefacts.

        Parameters
        ==========
        path
            Checkpoint directory.
        retain_columns
            Processed columns that must be retained with the checkpoint.
            This is intended for downstream methods that require access to
            exact training representations.
        package_source
            Whether to embed the input dataset.

            ``None`` chooses automatically: immutable remote sources are
            referenced, while local or ephemeral sources are packaged.

            ``True`` always packages the resolved input data.

            ``False`` never packages it.
        retain_example
            Save one processed example when available.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.data_source is None:
            source = DataSource()
        else:
            source = self.data_source

        if save_source_data is None:
            save_source_data = all([
                self.data_in is not None,
                not source.is_remote,
            ])

        if save_source_data and self.data_in is None:
            raise ValueError(
                "Cannot package the data source because this "
                "`DataPipeline` has not loaded any input data."
            )

        if save_transformed_columns is None or save_transformed_columns is False:
            save_transformed_columns = []
        elif isinstance(save_transformed_columns, (tuple, list)):
            save_transformed_columns = list(save_transformed_columns)
        elif not save_transformed_columns is True:
            raise ValueError(
                "`save_transformed_columns` must be None, True, False, tuple, list "
                f"but was {type(save_transformed_columns)}: {save_transformed_columns}"
            )
        else:
            pass

        if all([
            save_transformed_columns or len(save_transformed_columns) > 0,
            self.data_out is None,
        ]):
            raise ValueError(
                "Cannot retain processed columns because this "
                "DataPipeline has not processed any data."
            )

        if save_transformed_columns is True:
            save_transformed_columns = list(self.data_out.column_names)

        if self.data_out is not None:
            missing = [
                column
                for column in save_transformed_columns
                if column not in self.data_out.column_names
            ]
        else:
            missing = []

        if len(missing) > 0:
            raise KeyError(
                "Requested retained columns are absent from "
                "processed data: "
                + ", ".join(missing)
            )

        checkpoint_config = {
            "pipeline": self.to_config(),
            "source": source.to_config(),
            "artefacts": {
                "source_data": DATA_FILENAME if save_source_data else None,
                "transformed_data": TRANSFORMED_FILENAME if len(save_transformed_columns) > 0 else None,
                "example": (
                    EXAMPLE_FILENAME
                    if (
                        not discard_example_data
                        and self.data_out_example is not None
                    )
                    else None
                ),
            },
        }

        save_json(checkpoint_config, str(path / CONFIG_FILENAME))

        if save_source_data:
            (
                self.data_in
                .with_format(None)
                .to_parquet(str(path / DATA_FILENAME))
            )

        if len(save_transformed_columns) > 0:
            (
                self.data_out
                .with_format(None)
                .select_columns(save_transformed_columns)
                .to_parquet(str(path / TRANSFORMED_FILENAME))
            )

        if (
            not discard_example_data
            and self.data_out_example is not None
        ):
            (
                self.data_out_example
                .with_format(None)
                .to_parquet(str(path / EXAMPLE_FILENAME))
            )

    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        *,
        cache_dir: str | None = None,
    ) -> "DataPipeline":
        """Restore a saved pipeline and its available data artefacts."""
        path = Path(path)
        config = load_json(path / CONFIG_FILENAME)

        pipeline = DataPipeline.from_config(
            config["pipeline"],
            cache_dir=cache_dir,
        )
        pipeline.data_source = (
            DataSource.from_config(
                config.get("source")
            )
        )

        artefacts = config.get("artefacts", {})
        data_filename = artefacts.get("source_data")
        retained_filename = artefacts.get("transformed_data")
        example_filename = artefacts.get("example")

        if data_filename is not None:
            pipeline.data_in = autoload(
                path / data_filename,
                cache_dir=pipeline.cache_dir,
            )
        elif (
            pipeline.data_source is not None
            and pipeline.data_source.is_remote
        ):
            source = pipeline.data_source
            pipeline.data_in = autoload(
                pipeline.data_source.uri,
                cache=pipeline.cache_dir,
            )

        if retained_filename is not None:
            pipeline.data_out = autoload(
                path / retained_filename,
                cache_dir=pipeline.cache_dir,
            )

        if example_filename is not None:
            pipeline.data_out_example = autoload(
                path / example_filename,
                cache_dir=pipeline.cache_dir,
            )

        pipeline._data_loaded = pipeline.data_in is not None
        if (
            pipeline.data_out is not None
            or pipeline.data_out_example is not None
        ):
            pipeline._inspect_data_out()
        return pipeline

    # for compatibility
    def save_checkpoint(self, *args, **kwargs):
        return self.save(*args, **kwargs)

    @classmethod
    def load_checkpoint(cls, *args, **kwargs):
        return cls.load(*args, **kwargs)

    def dataloader(
        self,
        dataset=None,
        *,
        batch_size: int = 32,
        shuffle: bool = False,
        collators: ColumnCollator | Mapping[str, Callable] | None = None,
        **kwargs
    ):
        """Create a PyTorch DataLoader from processed data."""
        # optional torch dependencies
        from torch.utils.data import DataLoader

        from .collate import ColumnCollator

        if dataset is None:
            dataset = self.data_out

        if dataset is None:
            raise ValueError(
                "No processed dataset available. "
                "Provide `dataset` or call the pipeline first with pipeline(data)."
            )

        if isinstance(collators,  Mapping) or collators is None:
            collators = ColumnCollator(collators)
        if not isinstance(collators, ColumnCollator):
            raise ValueError(
                "If provided, `collators` must be dict or ColumnCollator, "
                f"but was {type(collators)}: {collators}"
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collators,
            **kwargs,
        )
