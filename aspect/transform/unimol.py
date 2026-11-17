"""Frozen Uni-Mol v1 molecular embeddings."""

from collections.abc import Callable, Iterable, Mapping
from functools import cache

from carabiner import print_err
import numpy as np

from .registry import register_function

UNIMOL_REPO: str = "dptech/Uni-Mol-Models"
UNIMOL_REVISION: str = "9f19c45c718192888a1c8a1c905f69f0755ea502"
UNIMOL_DIM: int = 512


def _import_unimol():
    try:
        import torch
        from unimol_tools import UniMolRepr
    except ImportError as e:
        raise ImportError(
            f"The `unimol` featurizer needs torch and unimol-tools ({e}). "
            "Try `pip install aspect-data[unimol]`."
        ) from e
    return torch, UniMolRepr


def _load_unimol(
    batch_size: int,
    remove_hs: bool,
    device: str,
    revision: str | None,
    cache_dir: str | None = None,
):
    """Resolve the active cache before looking up the memoized encoder."""
    from ..cache import hub_cache_dir
    return _load_unimol_cached(
        batch_size, remove_hs, device, revision, cache_dir or hub_cache_dir()
    )


@cache
def _load_unimol_cached(
    batch_size: int,
    remove_hs: bool,
    device: str,
    revision: str | None,
    cache_dir: str,
):
    """Load frozen Uni-Mol v1 weights once per configuration and cache."""
    from huggingface_hub import hf_hub_download

    _, UniMolRepr = _import_unimol()
    infix = "no" if remove_hs else "all"
    weight = f"mol_pre_{infix}_h_220816.pt" if remove_hs else f"mol_pre_{infix}_h_220816.pt"
    paths = {
        name: hf_hub_download(
            UNIMOL_REPO, 
            name, 
            revision=revision, 
            cache_dir=cache_dir,
        )
        for name in (weight, "mol.dict.txt")
    }
    encoder = UniMolRepr(
        data_type="molecule",
        model_name="unimolv1",
        batch_size=batch_size,
        remove_hs=remove_hs,
        use_cuda=device == "cuda",
        pretrained_model_path=paths[weight],
        pretrained_dict_path=paths["mol.dict.txt"],
    )
    # Avoid a new multiprocessing pool for every batch of SMILES.
    encoder.params["multi_process"] = False
    for parameter in encoder.model.parameters():
        parameter.requires_grad_(False)
    encoder.model.eval()
    return encoder


@register_function("unimol")
def UniMolEmbedding(
    conformers: str = "aspect",
    batch_size: int = 32,
    remove_hs: bool = False,
    device: str | None = None,
    revision: str | None = UNIMOL_REVISION,
) -> Callable:
    """512-d frozen Uni-Mol v1 CLS embeddings, averaged over conformers.

    By default, chain Aspect's `conformers` transform first:
    `("smiles", ["conformers", "unimol"])`. Each molecule's conformer
    embeddings are averaged. Alternatively, pass SMILES directly with
    `conformers="unimol"` to use Uni-Mol's RDKit conformer generation
    (one conformer, fixed seed 42, 2D fallback on embedding failure).
    This does not use a neural conformer generator.

    Weights load on first inference into the shared Aspect cache; the
    encoder is memoized in eval mode and does not compute gradients.
    Weights are pinned to a Hugging Face commit by default (`revision`).
    Missing or invalid rows return zero vectors (with a warning).
    """
    _import_unimol()
    from .deep_functions import get_device

    if conformers not in ("aspect", "unimol"):
        raise ValueError("conformers must be 'aspect' or 'unimol'.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    device = str(device or get_device())
    if device not in ("cpu", "cuda"):
        raise ValueError("Uni-Mol supports device='cpu' or 'cuda'.")

    def _unimol(
        data: Mapping[str, Iterable],
        input_column: str,
    ) -> np.ndarray:
        rows = list(data[input_column])
        out = np.zeros((len(rows), UNIMOL_DIM), dtype=np.float32)
        counts = np.zeros(len(rows), dtype=np.int64)
        pending, atoms, coordinates, owners = [], [], [], []

        def _flush():
            encoder = _load_unimol(batch_size, remove_hs, device, revision)
            inputs = pending if conformers == "unimol" else {
                "atoms": atoms,
                "coordinates": coordinates,
            }
            result = encoder.get_repr(inputs)
            embeddings = np.asarray(
                result["cls_repr"] if isinstance(result, dict) else result,
                dtype=np.float32,
            )
            if embeddings.shape != (len(owners), UNIMOL_DIM):
                raise RuntimeError(
                    f"Uni-Mol returned {embeddings.shape}, expected "
                    f"({len(owners)}, {UNIMOL_DIM})."
                )
            np.add.at(out, owners, embeddings)
            np.add.at(counts, owners, 1)
            pending.clear()
            atoms.clear()
            coordinates.clear()
            owners.clear()

        from rdkit import Chem
        if conformers == "unimol":
            for i, smiles in enumerate(rows):
                if not isinstance(smiles, str) or not smiles or Chem.MolFromSmiles(smiles) is None:
                    continue
                pending.append(smiles)
                owners.append(i)
                if len(owners) == batch_size:
                    _flush()
        else:
            periodic_table = Chem.GetPeriodicTable()
            for i, row in enumerate(rows):
                if row is None:
                    continue
                if isinstance(row, str):
                    raise TypeError(
                        "unimol embeds conformers, not SMILES: chain `conformers` "
                        "first, or set conformers='unimol'."
                    )
                numbers = np.asarray(row["atomic_numbers"])
                xyzs = np.asarray(row["coordinates"], dtype=np.float32)
                if xyzs.ndim != 3 or xyzs.shape[1:] != (len(numbers), 3):
                    raise ValueError("Expected coordinates [conformers, atoms, 3].")
                symbols = [periodic_table.GetElementSymbol(int(z)) for z in numbers]
                for xyz in xyzs:
                    atoms.append(symbols)
                    coordinates.append(xyz)
                    owners.append(i)
                    if len(owners) == batch_size:
                        _flush()
        if owners:
            _flush()
        present = counts > 0
        out[present] /= counts[present, None]
        n_null = int((~present).sum())
        if n_null:
            print_err(f"[WARN] unimol: {n_null} missing or invalid molecule(s) embedded as zero vectors.")
        return out

    return _unimol
