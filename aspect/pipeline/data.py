"""Featurization pipeline for tabular data."""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
import os

from carabiner import cast, print_err
import numpy as np
from numpy.typing import ArrayLike

from .. import app_name, __version__
from ..checkpoint_utils import load_checkpoint_file, save_json
from .io import AutoDataset
from ..package_data import CACHE_DIR
from ..serializing import Preprocessor
from ..typing import DataLike, FeatureLike, StrOrIterableOfStr

DEFAULT_BATCH_SIZE: int = 1024
X_KEY: str = f"{app_name}/v{__version__}/inputs"
Y_KEY: str = f"{app_name}/v{__version__}/labels"
CONTEXT_KEY: str = f"{X_KEY}:context"
DEFAULT_FORMAT: str = "numpy"


@dataclass
class ColumnPipeline:

    """Featurize a single column through a sequence of :class:`Preprocessor` steps.

    Steps are chained automatically: the output of step *i* becomes the input
    of step *i+1*.  Only the first step reads ``input_column`` directly; every
    subsequent step reads the previous step's ``output_column``.

    Parameters
    ==========
    input_column : str
        Name of the raw column to featurize.
    steps : list of Preprocessor
        Transformations to apply in order.

    Examples
    ========
    >>> from aspect.serializing import Preprocessor
    >>> cp = ColumnPipeline("score", [Preprocessor("log", "score")])
    >>> cp.input_column
    'score'
    >>> cp.output_column.startswith("aspect/")
    True
    >>> len(cp.steps)
    1
    """

    input_column: str
    steps: List[Preprocessor] = field(default_factory=list)

    def __post_init__(self):
        """Auto-chain step input/output columns.

        Examples
        ========
        >>> from aspect.serializing import Preprocessor
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x"), Preprocessor("log", "x")])
        >>> cp.steps[1].input_column == cp.steps[0].output_column
        True
        """
        current_col = self.input_column
        wired = []
        for step in self.steps:
            p = Preprocessor(name=step.name, input_column=current_col, kwargs=dict(step.kwargs))
            wired.append(p)
            current_col = p.output_column
        self.steps = wired

    @property
    def output_column(self) -> str:
        """The column name produced by the last step (or ``input_column`` if no steps).

        Examples
        ========
        >>> from aspect.serializing import Preprocessor
        >>> ColumnPipeline("raw", []).output_column
        'raw'
        >>> cp = ColumnPipeline("raw", [Preprocessor("identity", "raw")])
        >>> cp.output_column.startswith("aspect/")
        True
        """
        return self.steps[-1].output_column if self.steps else self.input_column

    def apply(
        self,
        dataset,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """Apply all steps to *dataset*, adding featurized columns.

        Parameters
        ==========
        dataset : datasets.Dataset
            Input HuggingFace dataset.
        batch_size : int
            Rows processed per batch.

        Returns
        =======
        datasets.Dataset
            Dataset with new feature columns appended.

        Examples
        ========
        >>> from datasets import Dataset
        >>> from aspect.serializing import Preprocessor
        >>> ds = Dataset.from_dict({"x": [1.0, 4.0, 9.0]})
        >>> cp = ColumnPipeline("x", [Preprocessor("log", "x")])
        >>> out = cp.apply(ds)
        >>> cp.output_column in out.column_names
        True
        """
        for step in self.steps:
            dataset = dataset.map(
                step,
                batched=True,
                batch_size=batch_size,
                desc=f"{step.name}({step.input_column})",
            )
        return dataset

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Only ``name`` and ``kwargs`` are stored for each step; ``input_column``
        is re-derived automatically by :meth:`__post_init__` on reconstruction.

        Examples
        ========
        >>> from aspect.serializing import Preprocessor
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        >>> d = cp.to_dict()
        >>> d["input_column"]
        'x'
        >>> d["steps"][0]["name"]
        'identity'
        >>> "input_column" not in d["steps"][0]
        True
        """
        return {
            "input_column": self.input_column,
            "steps": [{"name": s.name, "kwargs": dict(s.kwargs)} for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnPipeline":
        """Reconstruct from a dict produced by :meth:`to_dict`.

        Examples
        ========
        >>> from aspect.serializing import Preprocessor
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        >>> cp2 = ColumnPipeline.from_dict(cp.to_dict())
        >>> cp2.input_column, cp2.output_column == cp.output_column
        ('x', True)
        """
        steps = [
            Preprocessor(name=s["name"], input_column="_", kwargs=s.get("kwargs", {}))
            for s in data["steps"]
        ]
        return cls(input_column=data["input_column"], steps=steps)

    def to_file(self, filename: str) -> None:
        """Save to a JSON file.

        Examples
        ========
        >>> import tempfile, os, json
        >>> from aspect.serializing import Preprocessor
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        >>> with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        ...     fname = f.name
        >>> cp.to_file(fname)
        >>> with open(fname) as fh:
        ...     d = json.load(fh)
        >>> d["input_column"]
        'x'
        >>> os.unlink(fname)
        """
        save_json(self.to_dict(), filename)

    @classmethod
    def from_file(cls, filename: str) -> "ColumnPipeline":
        """Load from a JSON file.

        Examples
        ========
        >>> import tempfile, os
        >>> from aspect.serializing import Preprocessor
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        >>> with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        ...     fname = f.name
        >>> cp.to_file(fname)
        >>> cp2 = ColumnPipeline.from_file(fname)
        >>> cp2.input_column
        'x'
        >>> os.unlink(fname)
        """
        with open(filename) as f:
            return cls.from_dict(json.load(f))


@dataclass
class DataPipeline:

    """Featurization pipeline for tabular data backed by HuggingFace datasets.

    The pipeline is organized in three layers:

    1. **Column layer** — each :class:`ColumnPipeline` transforms one column
       through a chain of :class:`~aspect.serializing.Preprocessor` steps.
    2. **Group layer** — ``column_groups`` is a nested list; each inner list
       defines one output vector formed by concatenating the outputs of its
       :class:`ColumnPipeline` members.  The same column pipeline can appear in
       multiple groups.
    3. **Concatenation layer** — groups are emitted as ``X_KEY`` (single group)
       or ``X_KEY:0000``, ``X_KEY:0001``, … (multiple groups), plus ``Y_KEY``
       for labels.

    The full pipeline spec is JSON-serializable via :meth:`to_dict` /
    :meth:`from_dict`.

    Parameters
    ==========
    column_groups : list of list of ColumnPipeline
        Nested specification of input features.
    label_columns : list of str
        Columns to use as targets (concatenated into a single label vector).

    Examples
    ========
    >>> from aspect.serializing import Preprocessor
    >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
    >>> pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
    >>> pipe.label_columns
    ['y']
    >>> len(pipe.column_groups)
    1
    """

    column_groups: List[List[ColumnPipeline]]
    label_columns: List[str] = field(default_factory=list)
    _format: str = DEFAULT_FORMAT
    _format_kwargs: Optional[Mapping[str, Any]] = None
    _default_cache: Optional[str] = None

    def __post_init__(self):
        """Initialize runtime state not part of the serialized spec.

        Examples
        ========
        >>> from aspect.serializing import Preprocessor
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        >>> pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        >>> pipe.training_data is None
        True
        >>> pipe.input_shape is None
        True
        """
        if self._default_cache is None:
            self._default_cache = CACHE_DIR
        if self._format_kwargs is None:
            self._format_kwargs = {}
        self._input_training_data = None
        self.training_data = None
        self.input_shape = None
        self.output_shape = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_na(
        x: Mapping[str, Any],
        types: Mapping[str, str],
    ) -> Dict[str, Any]:
        """Replace ``None`` values with type-appropriate fill values.

        Examples
        ========
        >>> _fill = DataPipeline._fill_na
        >>> batch = {"a": [1, None, 3], "b": [None, "hi", None]}
        >>> types = {"a": "int64", "b": "string"}
        >>> out = _fill(batch, types)
        >>> out["a"]
        [1, 0, 3]
        >>> out["b"]
        [0, 'hi', 0]
        """
        for key in x:
            this_type = types[key]
            if this_type.startswith(("int", "uint")):
                fill_value = 0
            elif this_type.startswith("float"):
                fill_value = 0.
            elif this_type in ("string", "large_string"):
                fill_value = 0
            else:
                fill_value = 0
            x[key] = [fill_value if v is None else v for v in x[key]]
        return x

    def _unique_column_pipelines(self) -> List[ColumnPipeline]:
        """Deduplicated list of ColumnPipelines across all groups.

        Deduplication is by ``output_column`` so the same featurization is
        never computed twice even when a column appears in multiple groups.

        Examples
        ========
        >>> from aspect.serializing import Preprocessor
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        >>> pipe = DataPipeline(column_groups=[[cp], [cp]], label_columns=[])
        >>> len(pipe._unique_column_pipelines())
        1
        """
        seen = {}
        for group in self.column_groups:
            for cp in group:
                seen.setdefault(cp.output_column, cp)
        return list(seen.values())

    def _apply(
        self,
        data: DataLike,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache: Optional[str] = None,
    ):
        """Load *data*, featurize, and concatenate into output vectors.

        Returns a HuggingFace :class:`~datasets.Dataset` with columns
        ``X_KEY`` (or ``X_KEY:0000`` / ``X_KEY:0001`` / … for multiple groups)
        and ``Y_KEY``.

        Examples
        ========
        >>> import pandas as pd
        >>> from aspect.serializing import Preprocessor
        >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.1, 0.2, 0.3]})
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        >>> pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        >>> ds = pipe._apply(df)
        >>> X_KEY in ds.column_names
        True
        >>> Y_KEY in ds.column_names
        True
        """
        dataset = AutoDataset.load(data, cache=cache or self._default_cache)._dataset

        # collect all unique input columns (raw) needed across all groups
        raw_input_cols = sorted(set(
            cp.input_column for cp in self._unique_column_pipelines()
        ))
        label_cols = list(self.label_columns)

        # fill nulls on raw columns
        cols_to_keep = sorted(set(raw_input_cols + label_cols))
        dataset = dataset.select_columns(
            [c for c in cols_to_keep if c in dataset.column_names]
        )
        feature_types = {k: str(v.dtype) for k, v in dataset.features.items()}
        dataset = dataset.map(
            self._fill_na,
            fn_kwargs={"types": feature_types},
            batched=True,
            batch_size=batch_size,
            desc="Filling NaN values",
        )

        # apply each unique column pipeline once
        for cp in self._unique_column_pipelines():
            dataset = cp.apply(dataset, batch_size=batch_size)

        # concatenate each group into one output vector
        n_groups = len(self.column_groups)
        group_keys = [
            X_KEY if n_groups == 1 else f"{X_KEY}:{i:04d}"
            for i in range(n_groups)
        ]

        def _concat_group(batch, output_key, col_pipeline_outputs):
            arrays = [np.asarray(batch[col]) for col in col_pipeline_outputs]
            stacked = [a if a.ndim > 1 else a[:, np.newaxis] for a in arrays]
            batch[output_key] = np.concatenate(stacked, axis=-1)
            return batch

        def _concat_labels(batch, output_key, cols):
            arrays = [np.asarray(batch[col]) for col in cols]
            stacked = [a if a.ndim > 1 else a[:, np.newaxis] for a in arrays]
            batch[output_key] = np.concatenate(stacked, axis=-1)
            return batch

        for gkey, group in zip(group_keys, self.column_groups):
            out_cols = [cp.output_column for cp in group]
            dataset = dataset.map(
                _concat_group,
                fn_kwargs={"output_key": gkey, "col_pipeline_outputs": out_cols},
                batched=True,
                batch_size=batch_size,
                desc=f"Concatenating → {gkey}",
            )

        if label_cols:
            dataset = dataset.map(
                _concat_labels,
                fn_kwargs={"output_key": Y_KEY, "cols": label_cols},
                batched=True,
                batch_size=batch_size,
                desc=f"Concatenating labels → {Y_KEY}",
            )

        keep = group_keys + ([Y_KEY] if label_cols else [])
        dataset = dataset.select_columns(keep)

        return dataset.with_format(self._format, **self._format_kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        data: DataLike,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache: Optional[str] = None,
    ):
        """Featurize training data and record shapes.

        Stores the processed dataset in ``self.training_data`` and sets
        ``input_shape`` and ``output_shape`` from the first example.

        Parameters
        ==========
        data : DataLike
            Training data (DataFrame, dict, CSV path, HF dataset, or Hub ref).
        batch_size : int
            Rows per processing batch.
        cache : str, optional
            HuggingFace cache directory override.

        Returns
        =======
        datasets.Dataset
            Featurized dataset.

        Examples
        ========
        >>> import pandas as pd
        >>> from aspect.serializing import Preprocessor
        >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.1, 0.2, 0.3]})
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        >>> pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        >>> _ = pipe.fit_transform(df)
        >>> pipe.input_shape
        (1,)
        >>> pipe.output_shape
        (1,)
        """
        raw_dataset = AutoDataset.load(data, cache=cache or self._default_cache)._dataset
        self._input_training_data = raw_dataset
        self.training_data = self._apply(data, batch_size=batch_size, cache=cache)

        example = self.training_data.with_format("numpy")[0]
        n_groups = len(self.column_groups)
        if n_groups == 1:
            self.input_shape = np.asarray(example[X_KEY]).shape
        else:
            self.input_shape = tuple(
                np.asarray(example[f"{X_KEY}:{i:04d}"]).shape for i in range(n_groups)
            )
        if self.label_columns:
            self.output_shape = np.asarray(example[Y_KEY]).shape
        else:
            self.output_shape = None

        return self.training_data

    def transform(
        self,
        data: DataLike,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache: Optional[str] = None,
    ):
        """Featurize new data using the existing pipeline spec.

        Parameters
        ==========
        data : DataLike
            Data to transform.
        batch_size : int
            Rows per processing batch.
        cache : str, optional
            HuggingFace cache directory override.

        Returns
        =======
        datasets.Dataset
            Featurized dataset.

        Examples
        ========
        >>> import pandas as pd
        >>> from aspect.serializing import Preprocessor
        >>> df_train = pd.DataFrame({"x": [1.0, 2.0], "y": [0.1, 0.2]})
        >>> df_test  = pd.DataFrame({"x": [3.0, 4.0], "y": [0.3, 0.4]})
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        >>> pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        >>> _ = pipe.fit_transform(df_train)
        >>> out = pipe.transform(df_test)
        >>> X_KEY in out.column_names
        True
        """
        return self._apply(data, batch_size=batch_size, cache=cache)

    # ------------------------------------------------------------------
    # JSON serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the pipeline spec to a JSON-compatible dict.

        Only the pipeline *specification* is serialized (column groups,
        label columns, format settings) — not the fitted training data or
        shape information.

        Examples
        ========
        >>> from aspect.serializing import Preprocessor
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        >>> pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        >>> d = pipe.to_dict()
        >>> d["label_columns"]
        ['y']
        >>> len(d["column_groups"])
        1
        >>> d["column_groups"][0][0]["input_column"]
        'x'
        """
        return {
            "column_groups": [
                [cp.to_dict() for cp in group]
                for group in self.column_groups
            ],
            "label_columns": list(self.label_columns),
            "_format": self._format,
            "_format_kwargs": dict(self._format_kwargs or {}),
        }

    def to_file(self, filename: str) -> None:
        """Save the pipeline spec to a JSON file.

        Examples
        ========
        >>> import tempfile, os, json
        >>> from aspect.serializing import Preprocessor
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        >>> pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        >>> with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        ...     fname = f.name
        >>> pipe.to_file(fname)
        >>> with open(fname) as fh:
        ...     d = json.load(fh)
        >>> d["label_columns"]
        ['y']
        >>> os.unlink(fname)
        """
        save_json(self.to_dict(), filename)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataPipeline":
        """Reconstruct a pipeline from a dict produced by :meth:`to_dict`.

        Examples
        ========
        >>> from aspect.serializing import Preprocessor
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        >>> pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        >>> pipe2 = DataPipeline.from_dict(pipe.to_dict())
        >>> pipe2.label_columns
        ['y']
        >>> pipe2.column_groups[0][0].input_column
        'x'
        """
        data = dict(data)
        column_groups = [
            [ColumnPipeline.from_dict(cp) for cp in group]
            for group in data.pop("column_groups")
        ]
        return cls(column_groups=column_groups, **data)

    @classmethod
    def from_file(cls, filename: str) -> "DataPipeline":
        """Load a pipeline spec from a JSON file.

        Examples
        ========
        >>> import tempfile, os
        >>> from aspect.serializing import Preprocessor
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        >>> pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        >>> with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        ...     fname = f.name
        >>> pipe.to_file(fname)
        >>> pipe2 = DataPipeline.from_file(fname)
        >>> pipe2.label_columns
        ['y']
        >>> os.unlink(fname)
        """
        with open(filename) as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def checkpoint(self, checkpoint_dir: str) -> None:
        """Save the pipeline spec and processed training data to disk.

        Creates *checkpoint_dir* if it does not exist.  Writes:

        - ``pipeline-spec.json`` — full JSON spec (reconstructable via
          :meth:`from_file`)
        - ``input-data.hf`` — raw training dataset (if available)
        - ``training-data.hf`` — featurized training dataset (if available)

        Examples
        ========
        >>> import tempfile, os
        >>> from aspect.serializing import Preprocessor
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        >>> pipe = DataPipeline(column_groups=[[cp]], label_columns=[])
        >>> with tempfile.TemporaryDirectory() as d:
        ...     pipe.checkpoint(d)
        ...     os.path.exists(os.path.join(d, "pipeline-spec.json"))
        True
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.to_file(os.path.join(checkpoint_dir, "pipeline-spec.json"))
        if self._input_training_data is not None:
            self._input_training_data.save_to_disk(
                os.path.join(checkpoint_dir, "input-data.hf")
            )
        if self.training_data is not None:
            (
                self.training_data
                .with_format(None)
                .save_to_disk(os.path.join(checkpoint_dir, "training-data.hf"))
            )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: str,
        cache_dir: Optional[str] = None,
    ) -> "DataPipeline":
        """Restore a pipeline from a checkpoint directory or HF Hub path.

        Parameters
        ==========
        checkpoint : str
            Local directory or ``hf://owner/repo`` reference written by
            :meth:`checkpoint`.
        cache_dir : str, optional
            HuggingFace cache directory.

        Returns
        =======
        DataPipeline

        Examples
        ========
        >>> import tempfile, pandas as pd
        >>> from aspect.serializing import Preprocessor
        >>> df = pd.DataFrame({"x": [1.0, 2.0], "y": [0.1, 0.2]})
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        >>> pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        >>> _ = pipe.fit_transform(df)
        >>> with tempfile.TemporaryDirectory() as d:
        ...     pipe.checkpoint(d)
        ...     pipe2 = DataPipeline.load_checkpoint(d)
        >>> pipe2.label_columns
        ['y']
        >>> pipe2.training_data is not None
        True
        """
        pipeline_spec = load_checkpoint_file(
            checkpoint,
            filename="pipeline-spec.json",
            callback="json",
            none_on_error=False,
            cache_dir=cache_dir,
        )
        pipe = cls.from_dict(pipeline_spec)
        pipe._input_training_data = load_checkpoint_file(
            checkpoint,
            filename="input-data.hf",
            callback="hf-dataset",
            none_on_error=True,
            cache_dir=cache_dir,
        )
        training_data = load_checkpoint_file(
            checkpoint,
            filename="training-data.hf",
            callback="hf-dataset",
            none_on_error=True,
            cache_dir=cache_dir,
        )
        if training_data is not None:
            pipe.training_data = training_data.with_format(
                pipe._format, **pipe._format_kwargs
            )
            example = pipe.training_data.with_format("numpy")[0]
            col_names = pipe.training_data.column_names
            n_groups = len(pipe.column_groups)
            if n_groups == 1:
                if X_KEY in col_names:
                    pipe.input_shape = np.asarray(example[X_KEY]).shape
            else:
                pipe.input_shape = tuple(
                    np.asarray(example[f"{X_KEY}:{i:04d}"]).shape
                    for i in range(n_groups)
                    if f"{X_KEY}:{i:04d}" in col_names
                )
            if Y_KEY in col_names:
                pipe.output_shape = np.asarray(example[Y_KEY]).shape
        return pipe

    def make_dataloader(self, dataset, batch_size: int, shuffle: bool):
        """Wrap *dataset* in a framework-specific dataloader.

        Override this in subclasses to return a PyTorch ``DataLoader``, JAX
        iterator, etc.  The base implementation raises :exc:`NotImplementedError`.

        Examples
        ========
        >>> from aspect.serializing import Preprocessor
        >>> cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        >>> pipe = DataPipeline(column_groups=[[cp]], label_columns=[])
        >>> try:
        ...     pipe.make_dataloader(None, 32, True)
        ... except NotImplementedError:
        ...     print("not implemented")
        not implemented
        """
        raise NotImplementedError(
            "Override make_dataloader() in a subclass to produce a "
            "framework-specific dataloader."
        )


# ---------------------------------------------------------------------------
# Chemistry mixin (unchanged logic, corrected base class)
# ---------------------------------------------------------------------------

class ChemMixinBase(DataPipeline):

    """Mixin that adds chemistry-specific preprocessing to :class:`DataPipeline`."""

    smiles_column: str = "clean_smiles"
    common_fp_column: str = "tanimoto_nn_fp"
    tanimoto_column: str = "tanimoto_nn"

    @staticmethod
    def _featurizer_constructor(
        smiles_column: str,
        use_fp: bool = True,
        use_2d: bool = True,
        extra_featurizers: Optional[FeatureLike] = None,
        _allow_no_features: bool = False,
    ) -> List[Preprocessor]:
        featurizer = []
        if all([
            not use_fp,
            not use_2d,
            extra_featurizers is None or len(extra_featurizers) == 0,
            not _allow_no_features,
        ]):
            print_err("No featurizers defined for fingerprint. Setting `use_fp=True`.")
            use_fp = True
        if use_fp:
            featurizer.append(Preprocessor(
                name="morgan-fingerprint",
                input_column=smiles_column,
            ))
        if use_2d:
            featurizer.append(Preprocessor(
                name="descriptors-2d",
                input_column=smiles_column,
            ))
        if extra_featurizers is not None:
            featurizer += extra_featurizers
        if len(featurizer) == 0 and not _allow_no_features:
            raise ValueError("No features defined for fingerprint.")
        return featurizer

    @staticmethod
    def preprocess_data(
        data: Mapping[str, ArrayLike],
        structure_column: str,
        smiles_column: str,
        input_representation: str = "smiles",
    ) -> Dict[str, np.ndarray]:
        """Generate clean, canonical SMILES.

        Examples
        ========
        >>> d = {"struc": ["CCC", "CCO"]}  # doctest: +SKIP
        >>> out = ChemMixinBase.preprocess_data(d, "struc", ChemMixinBase.smiles_column)  # doctest: +SKIP
        >>> out[ChemMixinBase.smiles_column]  # doctest: +SKIP
        ['CCC', 'CCO']
        """
        from schemist.converting import convert_string_representation
        converted = convert_string_representation(
            strings=data[structure_column],
            input_representation=input_representation,
            output_representation="smiles",
        )
        if len(data[structure_column]) > 1 and not isinstance(data[structure_column], str):
            result = list(converted)
        else:
            result = [converted]
        data[smiles_column] = result
        return data

    @staticmethod
    def _get_max_sim(
        query: ArrayLike,
        references: ArrayLike,
        aggregator=np.max,
    ):
        a_n_b = np.sum(query[np.newaxis] * references, axis=-1, keepdims=True)
        sum_q = np.sum(query)
        similarities = a_n_b / (
            sum_q + np.sum(references, axis=-1) - np.sum(a_n_b, axis=-1)
        )[..., np.newaxis]
        return aggregator(similarities)

    @staticmethod
    def _get_nn_tanimoto(
        x: Mapping[str, ArrayLike],
        refs_data: Mapping[str, ArrayLike],
        results_column: str,
        _in_key: str,
        _sim_fn,
    ) -> Dict[str, np.ndarray]:
        query_fps = x[_in_key]
        refs = refs_data[_in_key]
        results = [_sim_fn(q, refs) for q in query_fps]
        x[results_column] = np.stack(results, axis=0)
        return x
