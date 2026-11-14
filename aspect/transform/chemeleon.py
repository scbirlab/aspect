"""Frozen CheMeleon molecular embeddings."""

from collections.abc import Callable, Iterable, Mapping
from functools import cache
import hashlib
import os
from pathlib import Path

from carabiner import print_err
import numpy as np

from .cleaning import DEFAULT_PH, clean_mols
from .registry import register_function

CHEMELEON_URL: str = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
CHEMELEON_SHA256: str = "c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651"
CHEMELEON_DIM: int = 2048


def _import_chemeleon_backend():
    try:
        import torch
        from chemprop import featurizers, nn
        from chemprop.data import BatchMolGraph
    except ImportError as e:
        raise ImportError(
            f"The `chemeleon` featurizer needs torch, chemprop>=2.2, and RDKit ({e}). "
            "Try `pip install aspect-data[chemeleon]`."
        ) from e
    return torch, featurizers, nn, BatchMolGraph


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as e:
        raise ValueError(f"{name} must be a number, got {value!r}.") from e


def _chemeleon_dir() -> Path:
    """Weights directory: `chemeleon/` in the shared aspect/providence cache.

    The cache is the CLI `--cache` / `DataPipeline(cache_dir=...)` while a
    pipeline runs, else `$PROVIDENCE_CACHE`, else the platform user cache
    (on Linux, under `$XDG_CACHE_HOME`). See `aspect.cache`.

    """
    from ..cache import resolve_cache
    return Path(resolve_cache()) / "chemeleon"


def _download(url: str, dest: Path, timeout: float) -> None:
    """Stream `url` to a unique temporary file beside `dest`, then rename atomically.

    `timeout` bounds each socket operation, so a stalled transfer fails
    rather than hanging while holding the lock. The temporary file is removed
    on any failure.

    """
    import shutil
    import tempfile
    from urllib.request import urlopen

    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f"{dest.name}.part.")
    try:
        with os.fdopen(fd, "wb") as f, urlopen(url, timeout=timeout) as r:
            shutil.copyfileobj(r, f, length=1 << 20)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _chemeleon_checkpoint() -> Path:
    """Download (once) and verify the CheMeleon message-passing weights.

    Single-flight: a `FileLock` beside the weights serializes download and
    verification across processes, and the file appears only by atomic rename.
    Locking across nodes needs a filesystem with working locks (e.g. Lustre
    `flock`, not `localflock`); otherwise point the cache (`--cache` or
    `$PROVIDENCE_CACHE`) at node-local storage.

    A cached file failing the SHA-256 check (e.g. truncated by an
    interrupted copy) is replaced once before raising.

    Environment
    ===========
    ASPECT_CHEMELEON_LOCK_TIMEOUT
        Seconds to wait for another process's download. Default: 600.
    ASPECT_CHEMELEON_DOWNLOAD_TIMEOUT
        Socket timeout in seconds for the download. Default: 60.

    """
    from filelock import FileLock, Timeout

    lock_timeout = _env_float("ASPECT_CHEMELEON_LOCK_TIMEOUT", 600.)
    download_timeout = _env_float("ASPECT_CHEMELEON_DOWNLOAD_TIMEOUT", 60.)
    ckpt_dir = _chemeleon_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "chemeleon_mp.pt"
    lock_path = str(path) + ".lock"
    try:
        with FileLock(lock_path, timeout=lock_timeout):
            for attempt in range(2):
                if not path.exists():
                    print_err(f"[INFO] chemeleon: downloading weights from {CHEMELEON_URL} to {path}")
                    _download(CHEMELEON_URL, path, timeout=download_timeout)
                digest = _sha256(path)
                if digest == CHEMELEON_SHA256:
                    return path
                if attempt == 0:
                    print_err(
                        f"[WARN] chemeleon: checksum mismatch for {path} "
                        f"(got {digest}); re-downloading."
                    )
                    path.unlink(missing_ok=True)
            raise OSError(
                f"chemeleon: checksum mismatch for {path} after re-download "
                f"(got {digest}, expected {CHEMELEON_SHA256})."
            )
    except Timeout as e:
        raise OSError(
            f"chemeleon: timed out after {lock_timeout} s waiting for {lock_path}. "
            "Another process may be downloading, or the lock is stuck on a "
            "filesystem without working locks; point the cache (--cache or "
            "PROVIDENCE_CACHE) at node-local storage or raise "
            "ASPECT_CHEMELEON_LOCK_TIMEOUT."
        ) from e


@cache
def _load_chemeleon(device: str):
    """Load (and memoize) the frozen CheMeleon encoder in eval mode."""
    torch, _, nn, _ = _import_chemeleon_backend()
    ckpt = torch.load(_chemeleon_checkpoint(), weights_only=True, map_location="cpu")
    mp = nn.BondMessagePassing(**ckpt["hyper_parameters"])
    mp.load_state_dict(ckpt["state_dict"])
    agg = nn.MeanAggregation()
    mp = mp.float().to(torch.device(device)).eval()
    for p in mp.parameters():
        p.requires_grad_(False)
    return mp, agg


@register_function("chemeleon")
def CheMeleon(
    batch_size: int = 256,
    device: str | None = None,
    desalt: bool = True,
    protonate: bool = True,
    protonation_pH: float = DEFAULT_PH,
) -> Callable:
    """Frozen CheMeleon foundation-model embeddings from SMILES.

    CheMeleon (Burns et al., 2025; https://doi.org/10.48550/arXiv.2506.15792)
    is a Chemprop D-MPNN pre-trained to predict Mordred descriptors. This
    transform runs its message-passing encoder with fixed weights (eval mode,
    fp32, `torch.no_grad()`) and mean-aggregates atom embeddings, matching
    upstream `CheMeleonFingerprint`.

    Weights are pinned to Zenodo record 15460715 and verified by SHA-256, so
    cached features stay valid. Weights are fetched on first call into the
    shared aspect/providence cache (see `aspect.cache`), so constructing the
    transform, e.g. when building a `DataPipeline`, downloads nothing. See
    `_chemeleon_checkpoint` for download locking and timeouts.

    Parameters
    ==========
    batch_size : int
        Molecules per forward pass. Graphs are not padded, so embeddings do
        not depend on batching.
    device : str, optional
        Torch device. Default: CUDA if available, else CPU.
    desalt : bool
        Embed only the parent fragment (schemist / RDKit `FragmentParent`),
        dropping counterions and solvents. Default: `True`.
    protonate : bool
        Set the dominant protonation state at `protonation_pH` (schemist /
        Dimorphite-DL). Default: `True`. With `desalt` and `protonate` both
        `False`, SMILES are parsed with plain `MolFromSmiles`, matching
        upstream exactly.
    protonation_pH : float
        pH for `protonate`. Default: 6.7.

    Returns
    =======
    Callable
        Function mapping (data, input_column) → float32 np.ndarray [N, 2048].
        Missing or unparseable SMILES get a zero vector (with a warning).

    """
    torch, featurizers, _, BatchMolGraph = _import_chemeleon_backend()
    from .deep_functions import get_device

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    device = str(device or get_device())
    # CheMeleon was trained on V2 atom features; pin it rather than relying
    # on chemprop's default.
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2(),
    )

    def _embed_batch(mols: list) -> np.ndarray:
        mp, agg = _load_chemeleon(device)  # memoized; loads on first use
        bmg = BatchMolGraph([featurizer(m) for m in mols])
        bmg.to(device=device)
        with torch.no_grad():
            H_v = mp(bmg)
            return agg(H_v, bmg.batch).float().cpu().numpy()

    def _chemeleon(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> np.ndarray:
        parsed = clean_mols(
            data[input_column],
            desalt=desalt,
            protonate=protonate,
            protonation_pH=protonation_pH,
        )
        out = np.zeros((len(parsed), CHEMELEON_DIM), dtype=np.float32)
        idx = [i for i, m in enumerate(parsed) if m is not None]
        mols = [parsed[i] for i in idx]
        n_invalid = len(parsed) - len(idx)
        if n_invalid > 0:
            print_err(
                f"[WARN] chemeleon: {n_invalid} missing or invalid SMILES "
                "embedded as zero vectors."
            )
        for start in range(0, len(mols), batch_size):
            stop = start + batch_size
            out[idx[start:stop]] = _embed_batch(mols[start:stop])
        return out

    return _chemeleon
