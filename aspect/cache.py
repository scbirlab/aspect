"""Cache location shared with providence.

Resolution order (providence `resolve_cache`): an explicit directory (e.g.
the CLI `--cache` or `DataPipeline(cache_dir=...)`), then
`$PROVIDENCE_CACHE`, then the platform user cache directory (on Linux,
`$XDG_CACHE_HOME/providence/<version>`, defaulting to `~/.cache`).

"""

from collections.abc import Iterator
from contextlib import contextmanager
import os

CACHE_ENV: str = "PROVIDENCE_CACHE"


def resolve_cache(cache: str | os.PathLike | None = None) -> str:
    """Absolute cache directory, created if missing.

    Examples
    ========
    >>> import os, tempfile
    >>> d = tempfile.mkdtemp()
    >>> resolve_cache(d) == os.path.abspath(d)
    True

    """
    from providence.package_data import resolve_cache as _resolve
    return _resolve(None if cache is None else os.fspath(cache))


def hub_cache_dir(cache: str | os.PathLike | None = None) -> str:
    """Hugging Face Hub cache inside the resolved cache, as providence lays it out.

    Passed explicitly as `cache_dir` to `from_pretrained`, because
    `huggingface_hub` reads `HF_HUB_CACHE` once, at import.

    Examples
    ========
    >>> import os, tempfile
    >>> d = tempfile.mkdtemp()
    >>> hub_cache_dir(d) == os.path.join(os.path.abspath(d), "hub")
    True

    """
    hub = os.path.join(resolve_cache(cache), "hub")
    os.makedirs(hub, exist_ok=True)
    return hub


@contextmanager
def use_cache(cache: str | os.PathLike | None = None) -> Iterator[str]:
    """Expose the resolved cache to transforms via `$PROVIDENCE_CACHE`, restoring it after.

    Transforms run inside Hugging Face Datasets `.map()`, possibly in worker
    processes, so the cache is passed through the environment rather than
    as a transform argument (which would also change output column hashes).

    Examples
    ========
    >>> import os, tempfile
    >>> d = tempfile.mkdtemp()
    >>> before = os.environ.get(CACHE_ENV)
    >>> with use_cache(d) as c:
    ...     os.environ[CACHE_ENV] == c == resolve_cache()
    True
    >>> os.environ.get(CACHE_ENV) == before
    True

    """
    resolved = resolve_cache(cache)
    previous = os.environ.get(CACHE_ENV)
    os.environ[CACHE_ENV] = resolved
    try:
        yield resolved
    finally:
        if previous is None:
            os.environ.pop(CACHE_ENV, None)
        else:
            os.environ[CACHE_ENV] = previous
