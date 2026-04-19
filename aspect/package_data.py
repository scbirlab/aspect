"""Tools for loading and writing reusable package data."""

import os

from platformdirs import user_cache_dir

from .. import app_name, __author__, __version__

CACHE_DIR = user_cache_dir(
    app_name,
    appauthor=False,
    version=__version__,
)

ELUENT_CACHE = "ELUENT_CACHE"
DEFAULT_CACHE = CACHE_DIR

def _get_data_path(
    filename: str, 
    env_key: str = ELUENT_CACHE,
    default: str = DEFAULT_CACHE
) -> str:
    """Returns the path to a writable version of a package data file.
    
    Copies it from the package resources if not present.

    """
    cache_dir = os.environ.get(
        env_key, 
        os.path.expanduser(default),
    )
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    return cache_dir, os.path.join(cache_dir, filename)
