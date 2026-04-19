"""Classes to enable JSON serialization of preprocessing functions."""

from typing import Any, Dict, Iterable, Mapping, Union
from dataclasses import asdict, dataclass, field
import json

import numpy as np

from . import app_name, __version__
from .checkpoint_utils import save_json
from .transform.registry import FUNCTION_REGISTRY
from . import functions as _functions  # noqa: populate FUNCTION_REGISTRY


@dataclass
class Preprocessor:

    name: str
    input_column: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            self.closure = FUNCTION_REGISTRY[self.name]
        except KeyError:
            raise ValueError(f"Function '{self.name}' is not registered.")
        self.hash = self._get_hash()
        self.hash_stub = self.hash[:8]
        self.output_column = self._get_output_column()
        self.function = self.closure(**self.kwargs)

    def _get_hash(self):
        from datasets.fingerprint import Hasher
        return Hasher.hash(self.to_dict())

    def _get_output_column(self):
        return f"{app_name}/{__version__}/feat:{self.input_column}:{self.name}={self.hash_stub}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict.

        Examples
        ========
        >>> p = Preprocessor(name="identity", input_column="x")
        >>> d = p.to_dict()
        >>> d["name"], d["input_column"]
        ('identity', 'x')
        >>> d["kwargs"]
        {}
        """
        return asdict(self)

    def to_file(self, filename: str) -> None:
        """Save to a JSON file.

        Examples
        ========
        >>> import tempfile, os, json
        >>> p = Preprocessor(name="identity", input_column="x")
        >>> with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        ...     fname = f.name
        >>> p.to_file(fname)
        >>> with open(fname) as f:
        ...     d = json.load(f)
        >>> d["name"]
        'identity'
        >>> os.unlink(fname)
        """
        return save_json(self.to_dict(), filename)

    @classmethod
    def from_dict(cls, data: Mapping[str, Union[str, Mapping]]) -> 'Preprocessor':
        """Reconstruct from a dict.

        Examples
        ========
        >>> p = Preprocessor.from_dict({"name": "identity", "input_column": "x", "kwargs": {}})
        >>> p.name, p.input_column
        ('identity', 'x')
        """
        return cls(**data)

    @classmethod
    def from_file(cls, filename: str) -> 'Preprocessor':
        """Load from a JSON file.

        Examples
        ========
        >>> import tempfile, os
        >>> p = Preprocessor(name="identity", input_column="col")
        >>> with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        ...     fname = f.name
        >>> p.to_file(fname)
        >>> p2 = Preprocessor.from_file(fname)
        >>> p2.name, p2.input_column
        ('identity', 'col')
        >>> os.unlink(fname)
        """
        with open(filename, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def show(cls) -> tuple:
        """List registered function names.

        Examples
        ========
        >>> "identity" in Preprocessor.show()
        True
        >>> "log" in Preprocessor.show()
        True
        """
        return tuple(FUNCTION_REGISTRY)

    def __call__(self, inputs: Mapping[str, Iterable]) -> Dict[str, np.ndarray]:
        """Execute the function with stored parameters.

        Examples
        ========
        >>> import numpy as np
        >>> p = Preprocessor(name="identity", input_column="x")
        >>> out = p({"x": [1.0, 2.0, 3.0]})
        >>> list(out["x"])
        [1.0, 2.0, 3.0]
        >>> p.output_column in out
        True
        """
        inputs[self.output_column] = self.function(inputs, self.input_column)
        return inputs
