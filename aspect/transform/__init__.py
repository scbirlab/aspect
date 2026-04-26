from .registry import FUNCTION_REGISTRY

def _load_all():
    try:
        from . import functions, deep_functions  # importing populates FUNCTION_REGISTRY as a side effect
    except (ImportError, NameError):
        pass

_load_all()