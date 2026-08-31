
from typing import TYPE_CHECKING, Any
from collections.abc import Iterable, Mapping
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

from .package_data import resolve_cache, configure_hf_cache


DATASETS_PREFIX: str = "hf://datasets/"

def hasher(
    s: str | bytes,
    n: int = 16,
) -> str:
    if isinstance(s, str):
        s = s.encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:n]


def _lock_path(
    key: str,
    cache_dir: str | None = None
) -> str:
    cache_dir = resolve_cache(cache_dir)
    locks_dir = os.path.join(cache_dir, ".locks")
    os.makedirs(locks_dir, exist_ok=True)
    h = hasher(key)
    return os.path.join(locks_dir, f"{h}.lock")


def _load_from_file(
    filename: str, 
    cache: str | None = None
) -> Dataset:

    cache, datasets_cache, _ = configure_hf_cache(cache)
    from datasets import load_dataset, Dataset, DatasetDict
    from filelock import FileLock

    filename = os.path.realpath(
        os.path.abspath(
            os.path.expanduser(filename)
        )
    )

    if filename.removesuffix(".gz").endswith((".csv", ".tsv", ".txt")):
        sep = "," if filename.endswith((".csv", ".csv.gz")) else "\t"
        read_f = partial(
            load_dataset,
            path="csv",
            data_files=filename,
            cache_dir=datasets_cache,
            sep=sep,
        )
        lock_key = "::".join([
            "file",
            "csv",
            filename,
            sep,
        ])
    elif filename.endswith((".arrow", ".hd5", ".json", ".parquet", ".xml")):
        _, ext = os.path.splitext(filename)
        protocol = ext.lstrip(".")
        read_f = partial(
            load_dataset,
            path=protocol,
            data_files=filename,
            cache_dir=datasets_cache,
        )
        lock_key = "::".join([
            "file",
            protocol,
            filename,
        ])
    elif filename.endswith(".hf"):
        ds = Dataset.load_from_disk(filename)
        if isinstance(ds, DatasetDict):
            return ds["train"]
        return ds
    else:
        raise IOError(f"Could not infer how to open '{filename}' from its extension.")

    # Cross-task lock on the shared filesystem
    lockfile = _lock_path(
        key=lock_key,
        cache_dir=cache, 
    )
    with FileLock(lockfile, timeout=60. * 60.):
        ds = read_f()

    if isinstance(ds, DatasetDict):
        return ds["train"]
    else:
        return ds


def _load_from_dataframe(
    dataframe: DataFrame | Mapping[str, ArrayLike],
    cache: str | None = None,
) -> Dataset:

    cache, datasets_cache, _ = configure_hf_cache(cache)
    from datasets import Dataset
    from filelock import FileLock
    from pandas import DataFrame
    from pandas.util import hash_pandas_object

    if not isinstance(dataframe, DataFrame):
        dataframe = DataFrame(dataframe)

    fingerprint = hashlib.sha256()
    fingerprint.update(
        repr([
            (str(col), str(dtype))
            for col, dtype in dataframe.dtypes.items()
        ]).encode()
    )
    fingerprint.update(
        dataframe.to_string(index=False).encode()
    )
    fingerprint = fingerprint.hexdigest()[:16]

    csv_dir = os.path.join(cache, "dataframes")
    csv_filename = f"{fingerprint}.parquet"
    csv_path = os.path.join(csv_dir, csv_filename)
    if os.path.exists(csv_path):
        return _load_from_file(
            filename=csv_path, 
            cache=datasets_cache,
        )

    lockfile = _lock_path(
        key=f"dataframe::{fingerprint}",
        cache_dir=csv_dir,
    )
    with FileLock(lockfile, timeout=60. * 60.):
        if os.path.exists(csv_path):
            return _load_from_file(
                filename=csv_path, 
                cache=datasets_cache,
            )
        dataframe.to_parquet(csv_path, index=False)
        ds = _load_from_file(
            filename=csv_path, 
            cache=datasets_cache,
        )

    return ds


def _get_ref_chunk(
    s, 
    sep: str | None = None, 
    all_seps: str = "@~:"
) -> str:
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
    cache: str | None = None
) -> Dataset:

    cache, datasets_cache, _ = configure_hf_cache(cache)
    from datasets import concatenate_datasets, load_dataset, DatasetDict
    from filelock import FileLock

    ref = ref.removeprefix(DATASETS_PREFIX).removeprefix("hf://")
    seps = "@~:"
    repo = _get_ref_chunk(ref, all_seps=seps)
    ver = _get_ref_chunk(ref, "@", all_seps=seps)
    split = _get_ref_chunk(ref, ":", all_seps=seps)
    config = _get_ref_chunk(ref, "~", all_seps=seps)
    
    lock_key = "::".join([
        "hf",
        repo,
        config or "",
        ver or "",
    ])
    lockfile = _lock_path(
        key=lock_key,
        cache_dir=cache,
    )

    with FileLock(
        lockfile,
        timeout=60 * 60,
    ):
        ds = load_dataset(
            path=repo, 
            name=config, 
            split=split, 
            revision=ver, 
            cache_dir=datasets_cache,
        )
    if isinstance(ds, DatasetDict):
        ds = concatenate_datasets([v for key, v in ds.items()])
    return ds


class AutoDataset:

    def __init__(self, dataset):
        self._dataset = dataset

    @classmethod
    def load(
        cls, 
        data: str | DataFrame, 
        cache: str | None = None
    ) -> Dataset | IterableDataset:
        from datasets import load_dataset, Dataset, IterableDataset
        from pandas import DataFrame

        if isinstance(data, (Dataset, IterableDataset)):
            dataset = data
        elif isinstance(data, (DataFrame, Mapping)):
            dataset = _load_from_dataframe(
                data, 
                cache=cache,
            )
        elif isinstance(data, str):
            if data.startswith("hf://"):
                dataset = _resolve_hf_hub_dataset(
                    data,
                    cache=cache,
                )
            elif os.path.exists(data):
                dataset = _load_from_file(
                    data,
                    cache=cache,
                )
            else:
                raise ValueError(
                    f"""
                    If `data` is a string, it must start with "{DATASETS_PREFIX}" or a path to an existing file. 
                    It was "{data}".
                    """
                )
        else:
            raise ValueError(
                """
                Data must be a string, Dataset, dictionary, or Pandas DataFrame.
                """
            )
        return cls(dataset)
