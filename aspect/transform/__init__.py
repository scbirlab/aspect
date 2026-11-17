from .registry import FUNCTION_REGISTRY

def _load_all():
    try:
        from . import (
            functions, 
            cleaning,
            chemprop,
            chemeleon,
            conformers,
            atomjepa,
            unimol,
            deep_functions 
        )  # importing populates FUNCTION_REGISTRY as a side effect
    except (ImportError, NameError) as e:
        raise e

_load_all()

from .base import ColumnTransform
