"""Tools to make allowed preprocessing functions."""

from typing import Callable, Dict

# Registry to map function names to actual implementations
FUNCTION_REGISTRY: Dict[str, Callable] = {}


def register_function(name: str):
    """Decorator to register a function in the function registry.
    
    """
    def decorator(f: Callable):             
        FUNCTION_REGISTRY[name] = f
        return f
        
    return decorator
