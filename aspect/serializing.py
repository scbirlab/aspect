"""Classes to enable JSON serialization of preprocessing functions."""

from typing import Any, Dict, Iterable, Mapping, Union
from dataclasses import asdict, dataclass, field

import numpy as np

from ... import app_name, __version__
from ...checkpoint_utils import save_json, _load_json
from .registry import FUNCTION_REGISTRY


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
        """Convert the function call to a JSON-serializable format."""
        return asdict(self)

    def to_file(self, filename: str) -> None:
        """Save to JSON."""
        return save_json(self.to_dict(), filename)

    @classmethod
    def from_dict(cls, data: Mapping[str, Union[str, Mapping]]) -> 'Preprocessor':
        """Reconstruct the function call from JSON."""
        return cls(**data)

    @classmethod
    def from_file(cls, filename: str) -> 'Preprocessor':
        """Load from JSON."""
        return cls.from_dict(_load_json(filename))

    @classmethod
    def show(cls) -> 'Preprocessor':
        """List registered functions."""
        return tuple(FUNCTION_REGISTRY)

    def __call__(self, inputs: Mapping[str, Iterable]) -> Dict[str, np.ndarray]:
        """Execute the function with stored parameters."""
        inputs[self.output_column] = self.function(inputs, self.input_column)
        return inputs
