
from typing import TYPE_CHECKING, Any
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
import json
import hashlib
import os

from carabiner import print_err

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from pandas import DataFrame
else:
    Dataset, DataFrame, IterableDataset = Any, Any, Any

from .package_data import resolve_cache, configure_hf_cache


DATASETS_PREFIX: str = "hf://datasets/"


def load_json(
    checkpoint: str, 
    filename: str | None = None
) -> dict[str, ...]:
    if filename is not None:
        path = os.path.join(checkpoint, filename)
    else:
        path = checkpoint
    with open(path, "r") as f:
        obj = json.load(f)
    return obj
   

def save_json(obj, filename: str) -> None:
    _dir = os.path.dirname(filename)
    if _dir != "." and len(_dir) > 0:
        os.makedirs(_dir, exist_ok=True)
    with open(filename, "w") as f:
        try:
            json.dump(obj, f, sort_keys=True, indent=4)
        except TypeError as e:
            print_err(f"{obj=}")
            raise e
    return None


def autoload(
    filename: str | os.PathLike,
    cache_dir: str | None = None
):
    return (
        AutoDataset
        .load(
            str(filename),
            cache=cache_dir,
        )
        ._dataset
    )


@dataclass(frozen=True)
class DataSource:
    """Provenance for a resolved dataset.

    Parameters
    ----------
    uri
        Original source URI or absolute local filename, when available.
    revision
        Immutable resolved source revision, when available.
    requested_revision
        Revision requested by the caller before resolution, when available.
    """

    uri: str | None = None
    requested_uri: str | None = None
    revision: str | None = None
    requested_revision: str | None = None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, str | None] | str
    ):
        if isinstance(config, str):
            if os.path.exists(config):
                config = load_json(config)
            else:
                raise FileNotFoundError(
                    "`config` was string, but filename "
                    f"called `{config}` not found."
                )

        config = deepcopy(dict(config))
        return cls(**config)

    def to_config(self, filename: str | None = None) -> dict[str, str | None]:
        config = asdict(self)
        if filename is not None:
            save_json(config, filename)
        return config

    @property
    def is_remote(self) -> bool:
        return all([
            self.uri is not None
            and self.uri.startswith((
                "hf://", 
                "https://", 
                "s3://",
            )),
            self.revision is not None,
        ])


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


def _resolve_hf_revision(
    repo: str,
    revision: str | None = None
) -> str:
    """Resolve a Hugging Face dataset revision to an immutable commit SHA."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(
        repo_id=repo,
        revision=revision,
    )
    return info.sha


def _resolve_hf_hub_dataset(
    ref: str, 
    cache: str | None = None
) -> Dataset:

    cache, datasets_cache, _ = configure_hf_cache(cache)
    from datasets import concatenate_datasets, load_dataset, DatasetDict
    from filelock import FileLock

    original_ref = ref
    ref = ref.removeprefix(DATASETS_PREFIX).removeprefix("hf://")
    seps = "@~:"
    repo = _get_ref_chunk(ref, all_seps=seps)
    requested_revision = _get_ref_chunk(ref, "@", all_seps=seps)
    split = _get_ref_chunk(ref, ":", all_seps=seps)
    config = _get_ref_chunk(ref, "~", all_seps=seps)

    revision = _resolve_hf_revision(
        repo=repo,
        revision=requested_revision,
    )
    
    lock_key = "::".join([
        "hf",
        repo,
        config or "",
        revision,
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
            revision=revision, 
            cache_dir=datasets_cache,
        )
    if isinstance(ds, DatasetDict):
        ds = concatenate_datasets([v for key, v in ds.items()])
    
    source = DataSource(
        uri=(
            f"{DATASETS_PREFIX}{repo}@{revision}" 
            + ('~' + config if config is not None else '')
            + (':' + split if split is not None else '')
        ),
        requested_uri=original_ref,
        revision=revision,
        requested_revision=requested_revision,
    )

    return ds, source


class AutoDataset:

    def __init__(
        self, 
        dataset: Dataset,
        source: DataSource | None = None
    ):
        self._dataset = dataset
        self.source = source or DataSource()

    @classmethod
    def load(
        cls, 
        data: str | DataFrame, 
        cache: str | None = None
    ) -> "AutoDataset":
        from datasets import load_dataset, Dataset, IterableDataset
        from pandas import DataFrame

        if isinstance(data, (Dataset, IterableDataset)):
            dataset = data
            source = None
        elif isinstance(data, (DataFrame, Mapping)):
            dataset = _load_from_dataframe(
                data, 
                cache=cache,
            )
            source = None
        elif isinstance(data, str):
            if data.startswith("hf://"):
                dataset, source = _resolve_hf_hub_dataset(
                    data,
                    cache=cache,
                )
            elif os.path.exists(data):
                filename = os.path.realpath(os.path.abspath(os.path.expanduser(data)))
                dataset = _load_from_file(
                    filename,
                    cache=cache,
                )
                source = DataSource(uri=filename)
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
        return cls(
            dataset=dataset,
            source=source,
        )
