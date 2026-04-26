"""Classes to enable JSON serialization of preprocessing functions."""

from typing import Any, Callable, Dict, Iterable, Mapping, Union
from dataclasses import asdict, dataclass, field
from functools import partial

from carabiner import print_err
import numpy as np

from .. import app_name, __version__
from ..checkpoint_utils import save_json, _load_json
from .registry import FUNCTION_REGISTRY


@dataclass
class ColumnTransform:

    """Serializable wrapper around a registered transform function.

    Examples
    ========
    Basic construction and output column naming:

    >>> ct = ColumnTransform(name="log", input_column="affinity")
    >>> ct.input_column
    'affinity'
    >>> ct.name
    'log'
    >>> ct.output_column.startswith("aspect/")
    True

    Output column is deterministic:

    >>> ct2 = ColumnTransform(name="log", input_column="affinity")
    >>> ct.output_column == ct2.output_column
    True

    Output column changes with input_column:

    >>> ct3 = ColumnTransform(name="log", input_column="other")
    >>> ct.output_column != ct3.output_column
    True

    Output column changes with kwargs:

    >>> ct4 = ColumnTransform(name="hash", input_column="assay", kwargs={"ndim": 64})
    >>> ct5 = ColumnTransform(name="hash", input_column="assay", kwargs={"ndim": 128})
    >>> ct4.output_column != ct5.output_column
    True

    Applying a transform:

    >>> import numpy as np
    >>> data = {"affinity": [1.0, 10.0, 100.0]}
    >>> result = ct(data)
    >>> ct.output_column in result
    True
    >>> np.allclose(result[ct.output_column], np.log([1., 10., 100.])[:,None])
    True

    Idempotency — calling twice skips recomputation:

    >>> result2 = ct(result)
    >>> np.array_equal(result[ct.output_column], result2[ct.output_column])
    True

    Serialization roundtrip:

    >>> d = ct.to_dict()
    >>> ct_restored = ColumnTransform.from_dict(d)
    >>> ct_restored.output_column == ct.output_column
    True
    >>> data2 = {"affinity": [1.0, 10.0, 100.0]}
    >>> result3 = ct_restored(data2)
    >>> np.allclose(result3[ct.output_column], np.log([1., 10., 100.])[:,None])
    True

    Chained application propagates columns correctly:

    >>> ct_hash = ColumnTransform(name="hash", input_column="assay", kwargs={"ndim": 16})
    >>> data3 = {"assay": ["MIC", "MIC", "binding"]}
    >>> r1 = ct_hash(data3)
    >>> ct_hash.output_column in r1
    True
    >>> r1[ct_hash.output_column].shape
    (3, 16)

    Unregistered name raises:

    >>> ColumnTransform(name="nonexistent", input_column="x")
    Traceback (most recent call last):
        ...
    ValueError: Function 'nonexistent' is not registered...

    show() lists registered functions:

    >>> transforms = ColumnTransform.show()
    >>> "log" in transforms
    True
    >>> "hash" in transforms
    True

    """

    name: Union[str, Callable]
    input_column: str
    kwargs: Mapping[str, Any] = field(default_factory = dict)
    _version: str = field(default=__version__)
    _app: str = field(default=app_name)

    def __post_init__(self):
        if isinstance(self.name, str):
            try:
                self.closure = FUNCTION_REGISTRY[self.name]
            except KeyError:
                raise ValueError(
                    f"Function '{self.name}' is not registered. Try: {', '.join(FUNCTION_REGISTRY)}"
                )
            self._serializable = True
        elif isinstance(self.name, Callable):
            self.closure = self.name
            self.name = str(self.name)
            self._serializable = False
        else:
            raise ValueError(f"Transformation function must be a str or callable")
        self.hash = self._get_hash()
        self.hash_stub = self.hash[:8]
        self.output_column = self._get_output_column()
        self.function = self.closure(**self.kwargs)

    def _get_hash(self):
        from datasets.fingerprint import Hasher
        return Hasher.hash(self.to_dict())

    def _get_output_column(self):
        return f"{self._app}/{self._version}/{self.name}:{self.input_column}/{self.hash_stub}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the function call to a JSON-serializable format."""
        if self._serializable:
            return asdict(self)
        else:
            raise AttributeError(f"Cannot serialize {self} with callable {self.name}.")

    def to_file(self, filename: str) -> None:
        """Save to JSON."""
        return save_json(self.to_dict(), filename)

    @classmethod
    def from_dict(cls, data: Mapping[str, Union[str, Mapping]]) -> 'ColumnTransform':
        """Reconstruct the function call from JSON."""
        return cls(**data)

    @classmethod
    def from_file(cls, filename: str) -> 'ColumnTransform':
        """Load from JSON."""
        return cls.from_dict(_load_json(filename))

    @classmethod
    def show(cls) -> 'ColumnTransform':
        """List registered functions."""
        return tuple(FUNCTION_REGISTRY)

    def apply(
        self, 
        inputs: Mapping[str, Iterable]
    ) -> np.ndarray:
        result = self.function(inputs, self.input_column)
        if result.ndim == 1:
            result = result[:, None]
        return result

    def __call__(
        self, 
        inputs: Mapping[str, Iterable]
    ) -> Dict[str, np.ndarray]:
        """Execute the function with stored parameters."""
        if self.output_column not in inputs:
            inputs[self.output_column] = self.apply(inputs)
        else:
            print_err(f"[WARN] {self.output_column} already present, skipping.")
        return inputs
