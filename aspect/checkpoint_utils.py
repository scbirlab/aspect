""""Utilities for loading and saving checkpoints."""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union
from tempfile import TemporaryDirectory
import os
import json

from carabiner import print_err

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
else:
    Dataset, IterableDataset = Any, Any


def _load_json(checkpoint: str, filename: str) -> Dict[str, Any]:
    with open(os.path.join(checkpoint, filename), "r") as f:
            obj = json.load(f)
    return obj
   

def save_json(obj, filename: str) -> None:
    with open(filename, "w") as f:
        try:
            json.dump(obj, f, sort_keys=True, indent=4)
        except TypeError as e:
            print_err(f"{obj=}")
            raise e
    return None


def _load_hf_dataset(checkpoint, filename) -> Union[Dataset, IterableDataset]:
    from datasets import load_from_disk
    return load_from_disk(os.path.join(checkpoint, filename))


def _load_torch_weights(checkpoint, filename):
    import torch
    return torch.load(
        os.path.join(checkpoint, filename),
        weights_only=True,
    )


FILE_LOADING_CALLBACKS = {
    "json": _load_json,
    "hf-dataset": _load_hf_dataset, 
    "pt": _load_torch_weights,
}


def load_checkpoint_file(
    checkpoint: str,
    filename: str,
    callback: Union[str, Callable] = "json",
    none_on_error: bool = False,
    cache_dir: Optional[str] = None,
    *args, **kwargs
) -> Union[Any, None]:
    from huggingface_hub import snapshot_download

    obj = None
    if isinstance(callback, str):
        try:
            callback = FILE_LOADING_CALLBACKS[callback.casefold()]
        except KeyError:
            raise ValueError(
                """
                File loading callback must be callable or name.
                """
            )
    if os.path.exists(checkpoint):
        obj = callback(checkpoint, filename)
    elif checkpoint.startswith("hf://"):
        checkpoint = checkpoint.split("hf://")[-1]
        if filename.endswith(".hf"):
            filename_pattern = [filename + '/*.arrow', filename + '/*.json']
        else:
            filename_pattern = filename
        with TemporaryDirectory() as tmpdirname:
            try:
                print_err(f"Looking up: {checkpoint} :: {filename}")
                snapshot_download(
                    repo_id=checkpoint,
                    allow_patterns=filename_pattern,
                    local_dir=tmpdirname,
                    cache_dir=cache_dir,
                    *args, **kwargs
                )
            except Exception as e:
                print_err(e)
                if none_on_error:
                    return None
                else:
                    raise e
            else:
                obj = callback(tmpdirname, filename)
    if obj is not None:
        return obj
    else:
        raise AttributeError(
            f"Could not load anything from {checkpoint=}, {filename=} with {callback=}."
        )
