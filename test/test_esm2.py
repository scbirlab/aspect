"""ESM2 protein embeddings."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

MODEL = "facebook/esm2_t6_8M_UR50D"
SHORT = "MKTAYIAKQRQISFVKSHFSRQ"
LONG = (
    "MSLLTEVETPIRNEWGCRCNDSSDPLVVAASIIGILHLILWILDRLFFKCIYRFFEHGLKRGPSTEG"
    "VPESMREEYRKEQQSAVDADDSHFVSIELE"
)


@pytest.fixture(scope="module")
def esm2():
    from aspect.transform.deep_functions import ESM2
    try:
        return ESM2(model=MODEL)
    except OSError as e:  # no network / hub unavailable
        pytest.skip(f"Could not load {MODEL}: {e}")


def _embed(f, seqs):
    return f({"seq": seqs}, "seq")


def test_shape(esm2):
    x = _embed(esm2, [SHORT, LONG, SHORT])
    assert x.shape == (3, 320)
    assert x.dtype == np.float32


def test_deterministic(esm2):
    a = _embed(esm2, [SHORT, LONG])
    b = _embed(esm2, [SHORT, LONG])
    assert np.array_equal(a, b)


def test_padding_invariance(esm2):
    alone = _embed(esm2, [SHORT])[0]
    batched = _embed(esm2, [SHORT, LONG])[0]
    assert np.allclose(alone, batched, atol=1e-5, rtol=0)


def test_invalid_sequences_are_zero(esm2):
    x = _embed(esm2, [SHORT, "", None, "MK-V"])
    assert np.any(x[0] != 0)
    assert np.all(x[1:] == 0)


def test_cls_pooling_differs_from_mean():
    from aspect.transform.deep_functions import ESM2
    try:
        f = ESM2(model=MODEL, pooling="cls")
    except OSError as e:
        pytest.skip(f"Could not load {MODEL}: {e}")
    mean = _embed(ESM2(model=MODEL), [SHORT])
    assert not np.allclose(_embed(f, [SHORT]), mean)


def test_long_sequences(esm2):
    from aspect.transform.deep_functions import ESM2, ESM2_MAX_RESIDUES
    long_seq = (LONG * 20)[:ESM2_MAX_RESIDUES + 300]
    head = long_seq[:ESM2_MAX_RESIDUES]
    truncated = _embed(esm2, [long_seq])
    assert np.allclose(truncated, _embed(esm2, [head]), atol=1e-5, rtol=0)
    chunked = _embed(ESM2(model=MODEL, long_sequences="chunk"), [long_seq])
    assert chunked.shape == (1, 320)
    assert not np.allclose(chunked, truncated)


def test_pipeline_registration():
    from aspect.data import DataPipeline
    try:
        pipe = DataPipeline({"emb": ("seq", {"name": "esm2", "kwargs": {"model": MODEL}})})
    except OSError as e:
        pytest.skip(f"Could not load {MODEL}: {e}")
    out = pipe({"seq": [SHORT, LONG]}).data
    assert np.asarray(out["emb"]).shape == (2, 320)
