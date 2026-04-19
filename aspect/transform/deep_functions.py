"""Preprocessing functions from deep embeddings."""

from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Mapping, Tuple
from functools import partial

from carabiner import cast, print_err
import numpy as np
from numpy.typing import ArrayLike
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, PreTrainedModel
else:
    PreTrainedTokenizerBase, PreTrainedModel = Any, Any

from .registry import register_function
from ..typing import StrOrIterableOfStr

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _index_into(x: ArrayLike, i: int):
    return x[:,i,:]


def _get_value_from_tuple(f: Callable, **kwargs):
    partial_f = partial(f, **kwargs)

    def _f(*args, **fkwargs):
        return partial_f(*args, **fkwargs).values

    return _f


EMBEDDING_AGGREGATORS = {
    "start": partial(_index_into, i=0),
    "end": partial(_index_into, i=-1),
    "sum": partial(torch.sum, dim=-2),
    "mean": partial(torch.mean, dim=-2),
    "median": _get_value_from_tuple(torch.median, dim=-2),
    "max": _get_value_from_tuple(torch.max, dim=-2),
}
DEFAULT_AGGREGATOR = EMBEDDING_AGGREGATORS["mean"]


def _aggregated_embedding(
    x: torch.Tensor, 
    aggregators: Iterable[Callable] = (DEFAULT_AGGREGATOR, )
):
    aggregated = [
        a(x).detach().cpu().numpy() for a in aggregators
    ]

    return np.concatenate(aggregated, axis=-1)


def _resolve_bart_model(ref: str) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(ref)
    return AutoTokenizer.from_pretrained(ref), torch.compile(model, fullgraph=True).to(_DEVICE)


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
        key: tokenized[key].to(_DEVICE) 
        for key in ['input_ids', 'attention_mask']
    }


@register_function("hf-bart")
def HfBART(
    ref: str,
    aggregator: StrOrIterableOfStr = "mean"
) -> np.ndarray:

    aggregators = [a.casefold() for a in cast(aggregator, to=list)]
    incorrect_agg = [
        a for a in aggregators if a not in EMBEDDING_AGGREGATORS
    ]
    if len(incorrect_agg) > 0:
        raise ValueError(f"Requested aggregators do not exist: {incorrect_agg}")
    else:
        aggregators = [EMBEDDING_AGGREGATORS[a] for a in aggregators]
    tokenizer, model = _resolve_bart_model(ref)

    def _hf_bart(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> np.ndarray:
        tokenized_inputs = _tokenize_for_embedding(
            data[input_column],
            tokenizer=tokenizer,
        )
        model.eval()
        with torch.no_grad():
            outputs = model(
                **tokenized_inputs, 
                decoder_input_ids=tokenized_inputs['input_ids'],
                output_hidden_states=True,
                return_dict=True,
            )
        enc_last = _aggregated_embedding(
            outputs.encoder_hidden_states[-1],
            aggregators=aggregators,
        )
        dec_last = _aggregated_embedding(
            outputs.decoder_hidden_states[-1],
            aggregators=aggregators,
        )
        return np.concatenate([enc_last, dec_last], axis=-1)

    return _hf_bart
