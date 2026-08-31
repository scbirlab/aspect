"""Tools for loading and writing reusable package data."""

import os

from platformdirs import user_cache_dir

from . import app_name, __version__


CACHE_DIR = user_cache_dir(
    app_name,
    version=__version__,
    appauthor=False,
)

APP_CACHE = "ASPECT_CACHE"
DEFAULT_CACHE = CACHE_DIR


def resolve_cache(cache: str | None = None) -> str:
    cache_dir = cache or os.environ.get(APP_CACHE, DEFAULT_CACHE)
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_data_path(
    filename: str,
    env_key: str = APP_CACHE,
    default: str = DEFAULT_CACHE,
) -> str:
    cache_dir = resolve_cache(
        os.environ.get(env_key, default)
    )
    configure_hf_cache(cache_dir)
    return cache_dir, os.path.join(cache_dir, filename)


def configure_hf_cache(
    cache: str | None = None,
) -> tuple[str, str, str]:
    cache_root = resolve_cache(cache)

    datasets_cache = os.path.join(cache_root, "datasets")
    hub_cache = os.path.join(cache_root, "hub")

    os.makedirs(datasets_cache, exist_ok=True)
    os.makedirs(hub_cache, exist_ok=True)

    os.environ["HF_HOME"] = cache_root
    os.environ["HF_DATASETS_CACHE"] = datasets_cache
    os.environ["HF_HUB_CACHE"] = hub_cache

    return cache_root, datasets_cache, hub_cache
