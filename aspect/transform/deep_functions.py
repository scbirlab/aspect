"""Preprocessing functions from deep embeddings."""

from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Mapping, Tuple
from functools import cache, partial

from carabiner import cast, print_err
import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, PreTrainedModel
else:
    PreTrainedTokenizerBase, PreTrainedModel = Any, Any

from .registry import register_function
from ..typing import StrOrIterableOfStr

def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@cache
def _resolve_bart_model(
    ref: str,
    device: str
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load (and memoize) a seq2seq tokenizer and model in eval mode.

    Not wrapped in `torch.compile`: with transformers >= 5 the output-capturing
    hooks are not graph-safe, so compilation fails on the first forward pass.

    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    model = AutoModelForSeq2SeqLM.from_pretrained(ref)
    return AutoTokenizer.from_pretrained(ref), model.to(torch.device(device)).eval()


def _tokenize_for_embedding(
    x: Iterable[str],
    tokenizer: PreTrainedTokenizerBase
) -> Dict[str, np.ndarray]:

    inputs = [
        _x if _x is not None 
        else '<unk>' 
        for _x in cast(x, to=list)
    ]
    tokenizer_args = {
        "return_tensors": "pt",
        "padding": True,
    }
    try:
        tokenized = tokenizer(inputs, **tokenizer_args)
    except (ValueError, TypeError) as e:
        print_err(e)
        print_err(f"Tokenizing this batch failed: {inputs}")
        tokenized = tokenizer(
            ['<unk>'] * len(inputs), 
            **tokenizer_args,
        )

    return {
        key: tokenized[key].to(get_device()) 
        for key in ['input_ids', 'attention_mask']
    }


def _masked_aggregators(torch) -> dict[str, Callable]:
    """Token aggregators over [B, T, H] hidden states that ignore padding.

    `mask` is [B, T] with 1 for real tokens and 0 for padding.

    """
    def _first(x, mask):
        idx = mask.argmax(dim=1)
        return x[torch.arange(x.shape[0], device=x.device), idx]

    def _last(x, mask):
        idx = mask.shape[1] - 1 - mask.flip(dims=(1, )).argmax(dim=1)
        return x[torch.arange(x.shape[0], device=x.device), idx]

    def _sum(x, mask):
        return (x * mask.unsqueeze(-1).to(x.dtype)).sum(dim=1)

    def _mean(x, mask):
        return _sum(x, mask) / mask.sum(dim=1, keepdim=True).to(x.dtype)

    def _median(x, mask):
        return x.masked_fill(mask.unsqueeze(-1) == 0, float("nan")).nanmedian(dim=1).values

    def _max(x, mask):
        return x.masked_fill(mask.unsqueeze(-1) == 0, float("-inf")).max(dim=1).values

    return {
        "start": _first,
        "end": _last,
        "sum": _sum,
        "mean": _mean,
        "median": _median,
        "max": _max,
    }


@register_function("hf-bart")
def HfBART(
    ref: str,
    aggregator: StrOrIterableOfStr = "mean",
    batch_size: int = 64,
    clean_smiles: bool = True,
    desalt: bool = True,
    protonate: bool = True,
    protonation_pH: float = 6.7
) -> Callable:
    """Encoder and decoder embeddings from a Hugging Face BART-style model.

    Parameters
    ==========
    ref : str
        Hugging Face seq2seq checkpoint.
    aggregator : str or list of str
        Token aggregation(s): "start", "end", "sum", "mean", "median", "max".
        Padding tokens are excluded, so results do not depend on batching.
    batch_size : int
        Maximum strings per forward pass. Inputs are length-sorted before
        batching to limit padding; this bounds memory regardless of how many
        rows the transform is handed.
    clean_smiles : bool
        Treat inputs as SMILES and canonicalize them before tokenizing, with
        `desalt` and `protonate` applied. Default: `True`. SMILES that fail
        to parse are embedded as `<unk>`. **Set `False` for non-SMILES text.**
    desalt : bool
        With `clean_smiles`, keep only the parent fragment (schemist / RDKit
        `FragmentParent`). Default: `True`.
    protonate : bool
        With `clean_smiles`, set the dominant protonation state at
        `protonation_pH` (schemist / Dimorphite-DL). Default: `True`.
    protonation_pH : float
        pH for `protonate`. Default: 6.7.

    Returns
    =======
    Callable
        Function mapping (data, input_column) → float32 np.ndarray
        [N, 2 * n_aggregators * hidden_size] (encoder, then decoder).

    """
    try:
        from transformers import PreTrainedTokenizerBase, PreTrainedModel
    except ImportError:
        raise ImportError("Hugging Face transformers not installed. Try `pip install aspect[deep]`.")
    else:
        import torch

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    embedding_aggregators = _masked_aggregators(torch)
    aggregators = [a.casefold() for a in cast(aggregator, to=list)]
    incorrect_agg = [
        a for a in aggregators if a not in embedding_aggregators
    ]
    if len(incorrect_agg) > 0:
        raise ValueError(f"Requested aggregators do not exist: {incorrect_agg}")
    aggregators = [embedding_aggregators[a] for a in aggregators]
    device = str(get_device())
    tokenizer, model = _resolve_bart_model(ref, device)
    width = 2 * len(aggregators) * model.config.d_model

    def _aggregated_embedding(x, mask) -> np.ndarray:
        return np.concatenate([
            a(x, mask).float().cpu().numpy() for a in aggregators
        ], axis=-1)

    def _embed_batch(inputs: list) -> np.ndarray:
        tokenized_inputs = _tokenize_for_embedding(
            inputs,
            tokenizer=tokenizer,
        )
        mask = tokenized_inputs["attention_mask"]
        with torch.no_grad():
            outputs = model(
                **tokenized_inputs,
                decoder_input_ids=tokenized_inputs["input_ids"],
                decoder_attention_mask=mask,
                output_hidden_states=True,
                return_dict=True,
            )
        return np.concatenate([
            _aggregated_embedding(outputs.encoder_hidden_states[-1], mask),
            _aggregated_embedding(outputs.decoder_hidden_states[-1], mask),
        ], axis=-1)

    def _hf_bart(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> np.ndarray:
        inputs = list(data[input_column])
        if clean_smiles:
            from .cleaning import clean_smiles as _clean
            cleaned = _clean(
                inputs,
                desalt=desalt,
                protonate=protonate,
                protonation_pH=protonation_pH,
            )
            n_invalid = sum(c is None for c in cleaned)
            if n_invalid > 0:
                print_err(
                    f"[WARN] hf-bart: {n_invalid} missing or invalid SMILES "
                    "embedded as <unk>."
                )
            inputs = cleaned
        out = np.zeros((len(inputs), width), dtype=np.float32)
        order = np.argsort(
            [len(x) if isinstance(x, str) else 0 for x in inputs],
            kind="stable",
        )
        for start in range(0, len(order), batch_size):
            idx = order[start:(start + batch_size)]
            out[idx] = _embed_batch([inputs[i] for i in idx])
        return out

    return _hf_bart


ESM2_MAX_RESIDUES: int = 1022  # 1024-token training crop minus CLS and EOS
_ESM2_ALPHABET: frozenset = frozenset("ACDEFGHIKLMNPQRSTVWYXBUZO")


def _import_esm_backend():
    try:
        import torch
        from transformers import AutoTokenizer, EsmModel
    except ImportError as e:
        raise ImportError(
            f"The `esm2` featurizer needs torch and transformers ({e}). "
            "Try `pip install aspect-data[esm]`."
        ) from e
    return torch, AutoTokenizer, EsmModel


@cache
def _load_esm2(
    model: str,
    revision: str | None,
    device: str
):
    """Load (and memoize) an ESM2 tokenizer and fp32 encoder in eval mode."""
    torch, AutoTokenizer, EsmModel = _import_esm_backend()
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    encoder = EsmModel.from_pretrained(
        model,
        revision=revision,
        add_pooling_layer=False,
    )
    encoder = encoder.float().to(torch.device(device)).eval()
    return tokenizer, encoder


def _clean_protein_sequence(x: str | None) -> str | None:
    """Uppercase, strip whitespace and one trailing stop `*`; `None` if invalid.

    Examples
    ========
    >>> _clean_protein_sequence(" mkv*")
    'MKV'
    >>> _clean_protein_sequence("MK-V") is None
    True
    >>> _clean_protein_sequence("") is None, _clean_protein_sequence(None) is None
    (True, True)

    """
    if not isinstance(x, str):
        return None
    x = "".join(x.split()).upper()
    if x.endswith("*"):
        x = x[:-1]
    if len(x) == 0 or not set(x) <= _ESM2_ALPHABET:
        return None
    return x


def _chunk_sequence(
    x: str,
    window: int,
    overlap: int
) -> list[str]:
    """Overlapping windows covering `x`.

    Examples
    ========
    >>> _chunk_sequence("ABCDEFGHIJ", window=4, overlap=1)
    ['ABCD', 'DEFG', 'GHIJ']
    >>> _chunk_sequence("ABCDEFGHI", window=4, overlap=1)
    ['ABCD', 'DEFG', 'FGHI']
    >>> _chunk_sequence("ABC", window=4, overlap=1)
    ['ABC']

    """
    if len(x) <= window:
        return [x]
    stride = window - overlap
    starts = list(range(0, len(x) - window, stride)) + [len(x) - window]
    return [x[s:(s + window)] for s in starts]


@cache
def _warn_unpinned(model: str) -> None:
    # cached so it warns once per model, not once per batch
    print_err(
        f"[WARN] esm2: no `revision` pinned for {model}; "
        "embeddings may change if the checkpoint is updated."
    )


@register_function("esm2")
def ESM2(
    model: str = "facebook/esm2_t12_35M_UR50D",
    revision: str | None = None,
    layer: int = -1,
    pooling: str = "mean",
    batch_size: int = 8,
    device: str | None = None,
    long_sequences: str = "truncate",
    chunk_overlap: int = 256
) -> Callable:
    """ESM2 protein language model embeddings from amino acid sequences.

    Runs the encoder in eval mode, fp32, under `torch.no_grad()`.

    Parameters
    ==========
    model : str
        Hugging Face ESM2 checkpoint, e.g. `facebook/esm2_t6_8M_UR50D` (320-d),
        `facebook/esm2_t12_35M_UR50D` (480-d, default), or
        `facebook/esm2_t33_650M_UR50D` (1280-d).
    revision : str, optional
        Hugging Face commit hash. **Pin this for reproducibility**: without it
        the latest checkpoint revision is downloaded, which can change
        embeddings without changing the cache key.
    layer : int
        Hidden state to pool. 0 is the embedding layer; -1 (default) is the
        final (layer-normed) layer.
    pooling : str
        `"mean"` averages residue tokens only (CLS, EOS, and padding excluded).
        `"cls"` takes the CLS token.
    batch_size : int
        Sequences (or windows) per forward pass. Sequences are length-sorted
        before batching; padding does not change embeddings.
    device : str, optional
        Torch device. Default: CUDA if available, else CPU.
    long_sequences : str
        For sequences longer than 1022 residues. `"truncate"` (default) keeps
        the first 1022 residues and warns with a count. `"chunk"` embeds
        overlapping 1022-residue windows and averages window embeddings
        weighted by window length.
    chunk_overlap : int
        Window overlap in residues for `long_sequences="chunk"`.

    Returns
    =======
    Callable
        Function mapping (data, input_column) → float32 np.ndarray
        [N, hidden_size]. Sequences are uppercased and stripped of whitespace
        and a trailing `*`. Missing, empty, or sequences containing characters
        outside the 25-letter ESM amino acid alphabet get a zero vector (with
        a warning).

    """
    torch, _, _ = _import_esm_backend()

    pooling = pooling.casefold()
    if pooling not in ("mean", "cls"):
        raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}.")
    if long_sequences not in ("truncate", "chunk"):
        raise ValueError(
            f"long_sequences must be 'truncate' or 'chunk', got {long_sequences!r}."
        )
    if not 0 <= chunk_overlap < ESM2_MAX_RESIDUES:
        raise ValueError(
            f"chunk_overlap must be in [0, {ESM2_MAX_RESIDUES}), got {chunk_overlap}."
        )
    if revision is None:
        _warn_unpinned(model)
    device = str(device or get_device())
    tokenizer, encoder = _load_esm2(model, revision, device)
    hidden_size = encoder.config.hidden_size

    def _embed_batch(seqs: list[str]) -> np.ndarray:
        tokenized = tokenizer(
            seqs,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        with torch.no_grad():
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = outputs.hidden_states[layer]
        if pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            residue_mask = (
                (1 - tokenized["special_tokens_mask"]) * tokenized["attention_mask"]
            ).to(device=device, dtype=hidden.dtype).unsqueeze(-1)
            pooled = (hidden * residue_mask).sum(dim=1) / residue_mask.sum(dim=1)
        return pooled.float().cpu().numpy()

    def _esm2(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> np.ndarray:
        raw = list(data[input_column])
        out = np.zeros((len(raw), hidden_size), dtype=np.float32)
        windows, owners, weights = [], [], []
        n_invalid, n_long = 0, 0
        for i, x in enumerate(raw):
            seq = _clean_protein_sequence(x)
            if seq is None:
                n_invalid += 1
                continue
            if len(seq) > ESM2_MAX_RESIDUES:
                n_long += 1
                if long_sequences == "truncate":
                    seq = seq[:ESM2_MAX_RESIDUES]
            chunks = (
                _chunk_sequence(seq, ESM2_MAX_RESIDUES, chunk_overlap)
                if long_sequences == "chunk" else [seq]
            )
            for chunk in chunks:
                windows.append(chunk)
                owners.append(i)
                weights.append(len(chunk))
        if n_invalid > 0:
            print_err(
                f"[WARN] esm2: {n_invalid} missing or invalid sequence(s) "
                "embedded as zero vectors."
            )
        if n_long > 0:
            action = (
                f"truncated to {ESM2_MAX_RESIDUES} residues"
                if long_sequences == "truncate"
                else "embedded in overlapping chunks"
            )
            print_err(
                f"[WARN] esm2: {n_long} sequence(s) longer than "
                f"{ESM2_MAX_RESIDUES} residues {action}."
            )
        if len(windows) == 0:
            return out

        order = np.argsort([len(w) for w in windows], kind="stable")
        embeddings = np.zeros((len(windows), hidden_size), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            idx = order[start:(start + batch_size)]
            embeddings[idx] = _embed_batch([windows[j] for j in idx])

        owners = np.asarray(owners)
        weights = np.asarray(weights, dtype=np.float64)
        for i in np.unique(owners):
            in_seq = owners == i
            out[i] = np.average(
                embeddings[in_seq],
                axis=0,
                weights=weights[in_seq],
            ).astype(np.float32)
        return out

    return _esm2
