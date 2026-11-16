"""hf-bart: no torch.compile, bounded batches, padding-invariant aggregation."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

REF = "hf-internal-testing/tiny-random-BartForConditionalGeneration"
SHORT = "CCO"
LONG = "CC(C)Cc1ccc(cc1)C(C)C(=O)OCCCCCCCCCCN"
AGGREGATORS = ["start", "end", "sum", "mean", "median", "max"]


def _hf_bart(**kwargs):
    from aspect.transform.deep_functions import HfBART, _resolve_bart_model, get_device
    try:
        _resolve_bart_model(REF, str(get_device()))  # transforms load lazily
    except OSError as e:  # no network / hub unavailable
        pytest.skip(f"Could not load {REF}: {e}")
    return HfBART(ref=REF, **kwargs)


def _embed(f, seqs):
    return f({"s": seqs}, "s")


def test_forward_runs_and_shape():
    # Used to fail on first forward under transformers >= 5 (torch.compile).
    f = _hf_bart(aggregator=["mean", "max"])
    x = _embed(f, [SHORT, LONG, SHORT])
    assert x.shape[0] == 3
    assert x.dtype == np.float32
    assert np.array_equal(x[0], x[2])


@pytest.mark.parametrize("aggregator", AGGREGATORS)
def test_padding_invariance(aggregator):
    f = _hf_bart(aggregator=aggregator)
    alone = _embed(f, [SHORT])[0]
    batched = _embed(f, [SHORT, LONG])[0]
    assert np.allclose(alone, batched, atol=1e-6, rtol=0)


def test_batches_are_bounded_and_order_preserved():
    from aspect.transform.deep_functions import _resolve_bart_model, get_device

    seqs = [("C" * (i % 17 + 1)) + "O" for i in range(50)]
    f_small = _hf_bart(batch_size=8)
    _, model = _resolve_bart_model(REF, str(get_device()))
    sizes = []
    hook = model.register_forward_pre_hook(
        lambda module, args, kwargs: sizes.append(kwargs["input_ids"].shape[0]),
        with_kwargs=True,
    )
    try:
        small = _embed(f_small, seqs)
    finally:
        hook.remove()
    assert max(sizes) <= 8
    assert sum(sizes) == len(seqs)
    whole = _embed(_hf_bart(batch_size=len(seqs)), seqs)
    assert np.allclose(small, whole, atol=1e-6, rtol=0)


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        _hf_bart(batch_size=0)


def test_inputs_used_as_given():
    salt, parent = "CC(=O)O.[Na+]", "CC(=O)[O-]"
    a, b = _embed(_hf_bart(), [salt, parent])
    assert not np.allclose(a, b)


def test_chained_clean_smiles():
    pytest.importorskip("schemist")
    from aspect.transform.cleaning import CleanSmiles
    salt, parent = "CC(=O)O.[Na+]", "CC(=O)[O-]"  # acetate at pH 7.4
    f = _hf_bart()
    a, b = _embed(f, CleanSmiles()({"s": [salt, parent]}, "s"))
    assert np.allclose(a, b, atol=1e-6, rtol=0)
    # Unparseable SMILES become None, embedded as <unk>.
    x = _embed(f, CleanSmiles()({"s": ["nonsense", "CCO"]}, "s"))
    assert np.allclose(x[0], _embed(f, [None])[0], atol=1e-6, rtol=0)
