"""Data pipeline class."""

from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Iterable, Mapping
from copy import copy
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
from providence import DataSource, DataResolver

from . import app_name, __version__
from .transform import ColumnTransform
from .typing import DataLike, StrOrIterableOfStr


DEFAULT_BATCH_SIZE: int = 1024
DEFAULT_DATALOADER_BATCH_SIZE: int = 32
DEFAULT_FORMAT: str = "numpy"

SCHEMA_VERSION: int = 1
CONFIG_FILENAME: str = "config.json"
DATA_FILENAME: str = "data.pv"
TRANSFORMED_FILENAME: str = "transformed.pv"
EXAMPLE_FILENAME: str = "example.parquet"

# Packages whose code determines feature values but which Hugging Face
# Datasets cannot see when fingerprinting the featurizing `.map()`.
_FEATURE_BACKENDS: tuple[str, ...] = (
    "schemist",
    "dimorphite_dl",
    "rdkit",
    "descriptastorus",
    "vectome",
    "chemprop",
    "transformers",
    "torch",
)


def _backend_versions() -> dict[str, str]:
    """Installed versions of feature-computing backends.

    Passed to the featurizing `.map()` so that upgrading a backend
    changes the Hugging Face Datasets cache fingerprint, rather than
    silently reusing features computed with the old backend.
    Packages that are not installed are omitted.

    Examples
    ========
    >>> versions = _backend_versions()
    >>> set(versions) <= set(_FEATURE_BACKENDS)
    True
    >>> all(isinstance(v, str) for v in versions.values())
    True

    """
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for package in _FEATURE_BACKENDS:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


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
        if this_type is None:
            continue
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
    >>> data = {"affinity": [1., 10., 100.]}
    >>> out = p(data)
    >>> "log_affinity" in out.data.column_names
    True
    >>> values = np.asarray(out.data["log_affinity"])
    >>> values.shape
    (3,)
    >>> np.allclose(values, np.log([1., 10., 100.]))
    True

    drop_unused_columns removes input columns:

    >>> out2 = p(data, drop_unused_columns=True)
    >>> "affinity" not in out2.data.column_names
    True
    >>> "log_affinity" in out2.data.column_names
    True

    keep_extra_columns preserved when dropping:

    >>> data2 = {"affinity": [1., 10.], "label": [0, 1]}
    >>> out3 = p(data2, drop_unused_columns=True, keep_extra_columns=["label"])
    >>> "label" in out3.data.column_names
    True

    Missing input column raises:

    >>> p({"wrong_col": [1., 20]})
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
        from providence.package_data import CACHE_DIR
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
        self.cache_dir = cache_dir or CACHE_DIR
        self._data_loaded = False

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
            from providence.io import load_json
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
            from providence.io import save_json
            save_json(config, str(filename))
        return config

    def clone(self):
        return type(self).from_config(
            self.to_config(),
            cache_dir=self.cache_dir,
        )

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

    @staticmethod
    def _featurize(
        x: Mapping[str, ArrayLike],
        column_transforms: Mapping[str, dict],
        backend_versions: Mapping[str, str] | None = None
    ) -> dict[str, np.ndarray]:
        # `backend_versions` is unused here: it is part of the cache key.

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
        dataset: DataLike | DataResolver, 
        batch_size: int = DEFAULT_BATCH_SIZE,
        drop_unused_columns: bool = False,
        keep_extra_columns: Iterable[str] | None = None,
        cache_dir: str | None = None
    ):
        from providence import DataSource, DataResolver
        if isinstance(dataset, DataResolver):
            resolved = dataset
        else:
            resolved = DataResolver(
                dataset, 
                cache_dir=cache_dir or self.cache_dir,
            )

        if len(self.column_transforms) == 0:
            # fast path if empty pipeline
            self.data_in = resolved
            self.data_out = copy(resolved)
            self._data_loaded = True
            return self.data_out

        input_columns = sorted(set(
            seq[0].input_column 
            for _, seq in self.column_transforms.items()
        ))  # get only the input column for each branch
        output_columns = sorted(set(self.column_transforms))
        
        _check_column_presence(
            input_columns, 
            resolved.data,
        )
        if drop_unused_columns:
            if keep_extra_columns is None:
                extra_cols = []
            else:
                extra_cols = list(keep_extra_columns)
            extra_cols = list(
                set(extra_cols + self.columns_to_keep)
                .intersection(
                    resolved.data.column_names
                )
            )
            all_input_columns = list(set(input_columns + extra_cols))
            resolved.data = (
                resolved.data
                .select_columns(all_input_columns)
            )
            all_output_columns = output_columns + extra_cols
        else:
            all_input_columns = list(resolved.data.column_names)
            all_output_columns = all_input_columns + output_columns

        def _check_type(f):
            if hasattr(f, "dtype"):
                return f.dtype
            elif hasattr(f, "feature"):
                return _check_type(f.feature)
            else:
                return None

        data_out = (
            resolved
            .data
            .map(
                _fill_na,
                fn_kwargs={
                    "types": {
                        key: _check_type(f)
                        for key, f in resolved.data.info.features.items()
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
                    "backend_versions": _backend_versions(),
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
            # .map(
            #     self._unsqueeze,
            #     fn_kwargs={"columns": output_columns},
            #     batched=True,
            #     batch_size=batch_size,
            #     desc="Unsqueezing",
            # )
        )
        self.data_in = resolved
        self.data_out = DataResolver(
            data_out,
            cache_dir=self.cache_dir,
        )
        self._data_loaded = True
        return self.data_out

    def save(
        self,
        path: str | os.PathLike,
        *,
        save_transformed: bool = False,
        save_source_data: bool | None = None,
        discard_example_data: bool = False
    ) -> None:
        """Save the pipeline and required training-data artefacts.

        Parameters
        ==========
        path
            Checkpoint directory.
        package_source
            Whether to embed the input dataset.

            ``None`` chooses automatically: immutable remote sources are
            referenced, while local or ephemeral sources are packaged.

            ``True`` always packages the resolved input data.

            ``False`` never packages it.
        discard_example_data
            Normally, save one processed example when available. Set 
            discard_example_data=True to turn off.
        """
        from providence.io import save_json
        from providence import DataSource, DataResolver

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.data_in is None:
            source = None
        elif isinstance(self.data_in.data_source, DataSource):
            source = self.data_in.data_source
        else:
            # should never happen
            raise TypeError(
                "Data_in source exists but is not a DataSource. "
                f"Instead, it is {type(self.data_in)}: {self.data_in}"
            )

        if save_source_data is None and self.data_in is not None:
            save_source_data = not source.is_remote

        if save_source_data and self.data_in is None:
            raise ValueError(
                "Cannot package the data source because this "
                "`DataPipeline` has not loaded any input data."
            )

        if self.data_out is not None:
            missing = [
                column
                for column in self.data_in.data.column_names
                if column not in self.data_out.data.column_names
            ]
        else: 
            missing = []

        if len(missing) > 0:
            raise KeyError(
                "Requested retained columns are absent from "
                "processed data: "
                + ", ".join(map(str, missing))
            )

        save_transformed = self.data_out is not None and save_transformed
        checkpoint_config = {
            "schema_version": SCHEMA_VERSION,
            "pipeline": self.to_config(),
            "artefacts": {
                "source_checkpoint": DATA_FILENAME if self.data_in is not None else None,
                "transformed_checkpoint": TRANSFORMED_FILENAME if save_transformed else None,
            },
        }

        save_json(
            checkpoint_config, 
            str(path / CONFIG_FILENAME),
        )
        if self.data_in is not None:
            from providence import DataResolver
            if isinstance(self.data_in, DataResolver):
                self.data_in.save(
                    str(path / checkpoint_config["artefacts"]["source_checkpoint"]),
                    save_source_data=save_source_data,
                    discard_example_data=discard_example_data,
                )
            else:
                # should never happen
                raise TypeError(
                    "Data_in exists but is not DataResolver. "
                    f"Instead, it is {type(self.data_in)}: {self.data_in}."
                )
        if save_transformed:
            self.data_out.save(
                str(path / checkpoint_config["artefacts"]["transformed_checkpoint"]),
                save_source_data=True,
                discard_example_data=discard_example_data,
            )

    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        *,
        skip_call: bool = False,
        cache_dir: str | None = None,
    ) -> "DataPipeline":
        """Restore a saved pipeline and its available data artefacts."""
        from providence.io import load_json
        from providence import DataSource, DataResolver

        path = Path(path)
        config = load_json(path / CONFIG_FILENAME)
        loaded_schema_version = config["schema_version"]
        if loaded_schema_version != SCHEMA_VERSION:
            raise AttributeError(
                f"Checkpoint schema is version {loaded_schema_version} "
                f"but package is using version {SCHEMA_VERSION}"
            )

        pipeline = DataPipeline.from_config(
            config["pipeline"],
            cache_dir=cache_dir,
        )

        artefacts = config.get("artefacts", {})
        data_checkpoint = artefacts.get("source_checkpoint")
        retained_checkpoint = artefacts.get("transformed_checkpoint")
        example_filename = artefacts.get("example")
        
        if data_checkpoint is not None:
            pipeline.data_in = DataResolver.load(
                path / data_checkpoint,
                cache_dir=cache_dir,
            )
        else:
            pipeline.data_in = None
        if retained_checkpoint is not None:
            pipeline.data_out = DataResolver.load(
                path / retained_checkpoint,
                cache_dir=cache_dir,
            )
        if all([
            pipeline.data_out is None,
            pipeline.data_in is not None,
            not skip_call,
        ]):
            pipeline(pipeline.data_in)
        pipeline._data_loaded = pipeline.data_in is not None
        return pipeline

    def write_table(
        self,
        path: str | os.PathLike
    ):
        if self.data_out is None:
            raise AttributeError("No data to write from this pipeline. Try calling it on some data first.")

        table = self.data_out
        table.write_table(path)
        return None            


    # for compatibility
    def save_checkpoint(self, *args, **kwargs):
        return self.save(*args, **kwargs)

    @classmethod
    def load_checkpoint(cls, *args, **kwargs):
        return cls.load(*args, **kwargs)

    @property
    def collators(self) -> dict[str, Callable]:
        """Runtime collators required by transformed output columns."""
        from .transform.registry import COLLATOR_REGISTRY
        from .collate import resolve_collator

        out = {}
        for column, transforms in self.column_transforms.items():
            if len(transforms) == 0:
                continue
            final_transform = transforms[-1]
            try:
                collator_path = COLLATOR_REGISTRY[final_transform.name]
            except KeyError:
                continue

            out[column] = resolve_collator(collator_path)
        return out


    def _resolve_collators(
        self,
        collators: Mapping[str, Callable] | None = None
    ):
        from .collate import ColumnCollator

        resolved_collators = dict(self.collators)

        if isinstance(collators, Mapping):
            resolved_collators.update(dict(collators))
        elif collators is not None:
            raise ValueError(
                "If provided, `collators` must be dict, "
                f"but was {type(collators)}: {collators}"
            )
        return ColumnCollator(resolved_collators)

    def collate(
        self,
        batch,
        *,
        collators: Mapping[str, Callable] | None = None,
    ):
        """Convert a batch of dataset values to runtime model inputs."""

        if isinstance(batch, Mapping):
            if len(batch) == 0:
                return {}

            n = len(next(iter(batch.values())))
            batch = [
                {
                    column: values[i]
                    for column, values
                    in batch.items()
                }
                for i in range(n)
            ]
        return self._resolve_collators(collators)(batch)

    def _apply_output_format(self, dataset):
        """Apply `output_format` unless the dataset is already formatted.

        Without a format, Arrow rows come back as Python lists and floats,
        so float32 feature columns are collated as float64.

        Examples
        ========
        >>> p = DataPipeline({"h": ("cat", {"name": "hash", "kwargs": {"ndim": 4}})})
        >>> ds = p({"cat": ["a", "b"]}).data
        >>> p._apply_output_format(ds)[0]["h"].dtype
        dtype('float32')
        >>> p._apply_output_format(ds.with_format("arrow")).format["type"]
        'arrow'

        """
        if self.output_format is None:
            return dataset
        current = getattr(dataset, "format", None)
        if isinstance(current, Mapping) and current.get("type") is not None:
            return dataset
        return dataset.with_format(
            self.output_format,
            **self.output_format_opts,
        )

    def dataloader(
        self,
        dataset=None,
        *,
        batch_size: int = DEFAULT_DATALOADER_BATCH_SIZE,
        shuffle: bool = False,
        collators: Mapping[str, Callable] | None = None,
        **kwargs
    ):
        """Create a PyTorch DataLoader from processed data."""
        # optional torch dependencies
        from torch.utils.data import DataLoader

        if dataset is None:
            dataset = self.data_out.data

        if dataset is None:
            raise ValueError(
                "No processed dataset available. "
                "Provide `dataset` or call the pipeline first with pipeline(data)."
            )
        dataset = self._apply_output_format(dataset)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._resolve_collators(collators),
            **kwargs,
        )
