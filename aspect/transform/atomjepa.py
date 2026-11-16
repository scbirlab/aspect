"""Frozen Atom-JEPA molecular embeddings."""

from collections.abc import Callable, Iterable, Mapping
from functools import cache
import os

from carabiner import print_err
import numpy as np

from .registry import register_function

ATOM_JEPA_REPO: str = "atom-jepa/atom-jepa"
ATOM_JEPA_REVISION: str = "bf14d025fc864d689dbad1ac9c55b8b7054af11e"
ATOM_JEPA_MODELS: tuple[str, ...] = ("molecules", "bio")
ATOM_JEPA_DIM: int = 256
ATOM_JEPA_LAYERS: int = 8


def _import_atom_jepa():
    try:
        import torch
        from atom_jepa import AtomJEPA
    except ImportError as e:
        raise ImportError(
            f"The `atom-jepa` featurizer needs torch and atom-jepa ({e}). "
            "Try `pip install aspect-data[atom-jepa]`."
        ) from e
    return torch, AtomJEPA


def _load_atom_jepa(
    model: str,
    revision: str,
    device: str,
    precision: str,
    cache_dir: str | None = None
):
    """Frozen Atom-JEPA encoder, from the active cache (see `aspect.cache`).

    The cache is resolved on every call, so `use_cache` and `--cache` route
    downloads, and the memoized encoder is keyed on the cache directory.

    """
    from ..cache import hub_cache_dir
    return _load_atom_jepa_cached(model, revision, device, precision, cache_dir or hub_cache_dir())


@cache
def _load_atom_jepa_cached(
    model: str,
    revision: str,
    device: str,
    precision: str,
    cache_dir: str
):
    """Download (once) and load (memoized) a frozen Atom-JEPA encoder.

    Weights come from the Hugging Face Hub at a pinned commit;
    `huggingface_hub` handles locking, atomic writes and integrity checks.

    """
    from huggingface_hub import hf_hub_download

    _, AtomJEPA = _import_atom_jepa()
    for filename in ("config.json", "model.safetensors"):
        path = hf_hub_download(
            ATOM_JEPA_REPO,
            f"{model}/{filename}",
            revision=revision,
            cache_dir=cache_dir,
        )
    encoder = AtomJEPA.from_pretrained(
        os.path.dirname(path),
        device=device,
        precision=precision,
        cuequivariance=False,  # keep fp32 numerics independent of GPU kernels
        verbose=False,
    )
    if (encoder.embedding_dim, encoder.num_layers) != (ATOM_JEPA_DIM, ATOM_JEPA_LAYERS):
        raise RuntimeError(
            f"Atom-JEPA {model} at {revision} has {encoder.num_layers} blocks of "
            f"{encoder.embedding_dim}, but aspect expects {ATOM_JEPA_LAYERS} of "
            f"{ATOM_JEPA_DIM}."
        )
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder.eval()


@register_function("atom-jepa")
def AtomJEPAEmbedding(
    model: str = "bio",
    layers: str | int = "last",
    batch_size: int = 32,
    device: str | None = None,
    precision: str = "fp32",
    revision: str = ATOM_JEPA_REVISION,
) -> Callable:
    """Frozen Atom-JEPA embeddings of 3D conformers.

    Atom-JEPA (https://arxiv.org/abs/2610.08400) is a self-supervised,
    equivariant 3D encoder; Atom-JEPA-Bio adds a second pretraining stage
    with PCBA bioassays. Input is a `conformers` column, so chain it first:
    `("smiles", ["conformers", "atom-jepa"])` matches upstream Atom-JEPA on
    SMILES exactly with the same RDKit version (tested with RDKit 2026.09 and
    atom-jepa 0.4.0). Each molecule's conformers are embedded with fixed
    weights and averaged; averaging embeddings is not the same as averaging
    predictions over conformers.

    Weights are pinned to a Hugging Face commit (`revision`), so cached
    features stay valid. They are fetched on first call, so constructing the
    transform downloads nothing.

    Parameters
    ==========
    model : str
        "bio" (Atom-JEPA-Bio, default) or "molecules" (Atom-JEPA).
    layers : str or int
        "last" (final block after the final norm, 256 values), "all" (each of
        the 8 blocks, flattened to 2048 values), or a block number 1-8.
    batch_size : int
        Conformers per forward pass. Conformers are converted to tensors one
        forward pass at a time, so this bounds tensor memory; the batch's
        conformer rows themselves are held in memory as usual.
    device : str, optional
        Torch device. Default: CUDA if available, else CPU.
    precision : str
        "fp32" (default, reproducible) or "bf16" (CUDA only).
    revision : str
        Hugging Face commit of the weights.

    Returns
    =======
    Callable
        Function mapping (data, input_column) → float32 np.ndarray [N, D].
        Null conformers (missing or invalid SMILES, or failed conformer
        generation) give zero vectors; chain `missing` after `conformers` to
        tell them from real embeddings.

    """
    torch, _ = _import_atom_jepa()
    from .deep_functions import get_device

    if model not in ATOM_JEPA_MODELS:
        raise ValueError(f"model must be one of {ATOM_JEPA_MODELS}, got {model!r}.")
    if layers == "all":
        dim = ATOM_JEPA_DIM * ATOM_JEPA_LAYERS
    elif layers == "last" or (isinstance(layers, int) and 1 <= layers <= ATOM_JEPA_LAYERS):
        dim = ATOM_JEPA_DIM
    else:
        raise ValueError(f"layers must be 'last', 'all' or 1-{ATOM_JEPA_LAYERS}, got {layers!r}.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    device = str(device or get_device())

    def _samples(row: Mapping) -> list[dict]:
        if isinstance(row, str):
            raise TypeError(
                "atom-jepa embeds conformers, not SMILES: chain `conformers` "
                "first, e.g. ('smiles', ['conformers', 'atom-jepa'])."
            )
        atomic_numbers = torch.as_tensor(np.asarray(row["atomic_numbers"]), dtype=torch.long)
        coordinates = np.asarray(row["coordinates"], dtype=np.float32)
        return [
            {"atomic_numbers": atomic_numbers, "node_coordinates": torch.from_numpy(xyz)}
            for xyz in coordinates
        ]

    def _atom_jepa(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> np.ndarray:
        rows = list(data[input_column])
        out = np.zeros((len(rows), dim), dtype=np.float32)
        counts = np.zeros(len(rows), dtype=np.int64)
        pending, owners = [], []

        def _flush():
            encoder = _load_atom_jepa(model, revision, device, precision)
            emb = encoder.embed(pending, layers=layers, batch_size=batch_size)
            np.add.at(out, owners, emb.numpy().reshape(len(pending), dim))
            pending.clear()
            owners.clear()

        # Flush at exactly `batch_size` conformers: bounded memory, and the
        # same forward passes as upstream `embed`, so results match it exactly.
        for i, row in enumerate(rows):
            if row is None:
                continue
            for sample in _samples(row):
                pending.append(sample)
                owners.append(i)
                counts[i] += 1
                if len(pending) == batch_size:
                    _flush()
        if pending:
            _flush()

        n_null = int((counts == 0).sum())
        if n_null > 0:
            print_err(f"[WARN] atom-jepa: {n_null} null conformers embedded as zero vectors.")
        present = counts > 0
        out[present] /= counts[present, None]
        return out

    return _atom_jepa
