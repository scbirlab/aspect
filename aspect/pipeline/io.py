"""Dataset loading utilities backed by HuggingFace datasets."""

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Union
from functools import partial
import hashlib
import os
import tempfile

from carabiner import print_err

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from pandas import DataFrame
else:
    Dataset, DataFrame, IterableDataset = Any, Any, Any

from ..package_data import DEFAULT_CACHE


DATASETS_PREFIX: str = "hf://datasets/"


def hasher(s: Union[str, bytes], n: int = 16) -> str:
    """Return the first *n* hex characters of the SHA-256 hash of *s*.

    Parameters
    ==========
    s : str or bytes
        Input to hash.
    n : int
        Length of the returned hex string.

    Returns
    =======
    str

    Examples
    ========
    >>> h = hasher("hello")
    >>> len(h)
    16
    >>> h == hasher(b"hello")
    True
    >>> hasher("hello") != hasher("world")
    True
    """
    if isinstance(s, str):
        s = s.encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:n]


def _lock_path(cache_dir: str, key: str) -> str:
    locks_dir = os.path.join(cache_dir, ".locks")
    os.makedirs(locks_dir, exist_ok=True)
    h = hasher(key.encode("utf-8"))
    return os.path.join(locks_dir, f"{h}.lock")


def _load_from_file(filename: str, cache: Optional[str] = None) -> Dataset:
    from datasets import load_dataset, Dataset, DatasetDict
    from filelock import FileLock

    cache = cache or DEFAULT_CACHE
    if filename.removesuffix(".gz").endswith((".csv", ".tsv", ".txt")):
        read_f = partial(
            load_dataset,
            path="csv",
            data_files=filename,
            cache_dir=cache,
            sep="," if filename.endswith((".csv", ".csv.gz")) else "\t",
        )
        lock_key = ("csv", read_f.keywords.get('sep'))
    elif filename.endswith((".arrow", ".hd5", ".json", ".parquet", ".xml")):
        _, ext = os.path.splitext(filename)
        protocol = ext.lstrip(".")
        read_f = partial(
            load_dataset,
            path=protocol,
            data_files=filename,
            cache_dir=cache,
        )
        lock_key = (protocol, "")
    elif filename.endswith(".hf"):
        read_f = partial(
            Dataset.load_from_disk,
            dataset_path=filename,
        )
        lock_key = ("hf", "")
    else:
        raise IOError(f"Could not infer how to open '{filename}' from its extension.")

    lockfile = _lock_path(cache, "_".join(str(k) for k in lock_key))
    with FileLock(lockfile, timeout=60. * 60.):
        ds = read_f()
    if isinstance(ds, DatasetDict):
        return ds["train"]
    else:
        return ds


def _load_from_dataframe(
    dataframe: Union[DataFrame, Mapping[str, ArrayLike], Iterable[Mapping[str, ArrayLike]]],
    cache: Optional[str] = None,
) -> Dataset:
    from pandas import DataFrame

    if cache is None:
        cache = DEFAULT_CACHE
        print_err(f"Defaulting to cache: {cache}")
    if not isinstance(dataframe, DataFrame):
        dataframe = DataFrame(dataframe)

    hash_name = hasher(dataframe.to_string())
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_filename = os.path.join(tmpdir, f"{hash_name}.csv.gz")
        dataframe.to_csv(csv_filename, index=False)
        ds = _load_from_file(csv_filename, cache=cache)
    return ds


def _get_ref_chunk(
    s: str,
    sep: Optional[str] = None,
    all_seps: str = "@~:"
) -> Optional[str]:
    """Extract a chunk from a HuggingFace Hub reference string.

    Parameters
    ==========
    s : str
        Reference string, e.g. ``"owner/repo@v1.0~config:split"``.
    sep : str, optional
        Separator that precedes the chunk to extract.  ``None`` extracts the
        leading part (before any separator in *all_seps*).
    all_seps : str
        Characters that act as separators.

    Returns
    =======
    str or None
        The extracted chunk, or ``None`` when *sep* is not found in *s*.

    Examples
    ========
    >>> _get_ref_chunk("owner/repo@v1~cfg:split", "@")
    'v1'
    >>> _get_ref_chunk("owner/repo@v1~cfg:split", "~")
    'cfg'
    >>> _get_ref_chunk("owner/repo@v1~cfg:split", ":")
    'split'
    >>> _get_ref_chunk("owner/repo@v1~cfg:split")
    'owner/repo'
    >>> _get_ref_chunk("owner/repo", "@") is None
    True
    """
    if sep is not None:
        if sep in s:
            s = s.rpartition(sep)[-1]
        else:
            return None
    for _sep in all_seps:
        s = s.partition(_sep)[0]
    return s


def _resolve_hf_hub_dataset(
    ref: str,
    cache: Optional[str] = None
) -> Dataset:
    from datasets import concatenate_datasets, load_dataset, DatasetDict

    ref = ref.removeprefix(DATASETS_PREFIX).removeprefix("hf://")
    seps = "@~:"
    ver = _get_ref_chunk(ref, "@", all_seps=seps)
    split = _get_ref_chunk(ref, ":", all_seps=seps)
    config = _get_ref_chunk(ref, "~", all_seps=seps)

    ds = load_dataset(
        path=_get_ref_chunk(ref, all_seps=seps),
        name=config,
        split=split,
        revision=ver,
        cache_dir=cache,
    )
    if isinstance(ds, DatasetDict):
        ds = concatenate_datasets([v for key, v in ds.items()])
    return ds


class AutoDataset:

    """Factory for loading tabular data from many formats into a HuggingFace dataset.

    Supported *data* types:

    - :class:`datasets.Dataset` or :class:`datasets.IterableDataset` — passed through
    - :class:`pandas.DataFrame` or ``dict`` — converted via a temporary CSV
    - ``str`` ending in ``.csv``, ``.tsv``, ``.parquet``, ``.json``, ``.arrow``,
      ``.hf`` — loaded from disk
    - ``"hf://<repo>[~config][@version][:split]"`` — loaded from HuggingFace Hub

    Examples
    ========
    >>> from aspect.pipeline.io import AutoDataset
    >>> ds = AutoDataset.load({"x": [1, 2, 3], "y": [4, 5, 6]})
    >>> ds._dataset.column_names
    ['x', 'y']
    >>> len(ds._dataset)
    3
    """

    def __init__(self, dataset):
        self._dataset = dataset

    @classmethod
    def load(
        cls,
        data: Union[str, "DataFrame"],
        cache: Optional[str] = None,
    ) -> "AutoDataset":
        """Load *data* and return an :class:`AutoDataset` wrapping it.

        Parameters
        ==========
        data : DataLike
            Input data in any supported format.
        cache : str, optional
            HuggingFace datasets cache directory.

        Returns
        =======
        AutoDataset

        Examples
        ========
        >>> from aspect.pipeline.io import AutoDataset
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        >>> ds = AutoDataset.load(df)
        >>> sorted(ds._dataset.column_names)
        ['a', 'b']

        Loading from a plain dict:

        >>> ds2 = AutoDataset.load({"x": [10, 20]})
        >>> list(ds2._dataset["x"])
        [10, 20]

        Passing through an existing Dataset:

        >>> from datasets import Dataset
        >>> raw = Dataset.from_dict({"z": [7, 8, 9]})
        >>> ds3 = AutoDataset.load(raw)
        >>> ds3._dataset is raw
        True
        """
        from datasets import load_dataset, Dataset, IterableDataset
        from pandas import DataFrame

        if isinstance(data, (Dataset, IterableDataset)):
            dataset = data
        elif isinstance(data, (DataFrame, Mapping)):
            dataset = _load_from_dataframe(data, cache=cache)
        elif isinstance(data, str):
            if data.startswith("hf://"):
                dataset = _resolve_hf_hub_dataset(data, cache=cache)
            elif os.path.exists(data):
                dataset = _load_from_file(data, cache=cache)
            else:
                raise ValueError(
                    f'If `data` is a string it must start with "hf://" or be '
                    f'a path to an existing file. Got: "{data}".'
                )
        else:
            raise ValueError(
                "data must be a string, Dataset, dict, or Pandas DataFrame."
            )
        return cls(dataset)
