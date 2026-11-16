"""Frozen CheMeleon molecular embeddings."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("chemprop")

SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]


@pytest.fixture(scope="module")
def chemeleon():
    from aspect.transform.chemeleon import CheMeleon
    f = CheMeleon(device="cpu")
    try:
        f({"smiles": SMILES[:1]}, "smiles")  # weights load on first call
    except OSError as e:  # no network / Zenodo unavailable
        pytest.skip(f"Could not load CheMeleon: {e}")
    return f


def _embed(f, smiles):
    return f({"smiles": smiles}, "smiles")


def test_shape(chemeleon):
    x = _embed(chemeleon, SMILES)
    assert x.shape == (3, 2048)
    assert x.dtype == np.float32
    assert np.all(np.isfinite(x))


def test_deterministic(chemeleon):
    assert np.array_equal(_embed(chemeleon, SMILES), _embed(chemeleon, SMILES))


def test_batch_invariance(chemeleon):
    from aspect.transform.chemeleon import CheMeleon
    full = _embed(chemeleon, SMILES)
    one_at_a_time = _embed(CheMeleon(device="cpu", batch_size=1), SMILES)
    assert np.allclose(full, one_at_a_time, atol=1e-5, rtol=0)


def test_invalid_smiles_are_zero(chemeleon):
    x = _embed(chemeleon, [SMILES[0], "", None, "not-a-smiles"])
    assert np.any(x[0] != 0)
    assert np.all(x[1:] == 0)


def test_smiles_used_as_given(chemeleon):
    salt, parent = "CC(=O)O.[Na+]", "CC(=O)[O-]"
    a, b = _embed(chemeleon, [salt, parent])
    assert not np.allclose(a, b)


def test_chained_clean_smiles(chemeleon):
    from aspect.transform.cleaning import CleanSmiles
    salt, parent = "CC(=O)O.[Na+]", "CC(=O)[O-]"  # acetate at pH 7.4
    cleaned = CleanSmiles()({"s": [salt, "CC(=O)O", parent]}, "s")
    a, b, c = _embed(chemeleon, cleaned)
    assert np.allclose(a, c, atol=1e-6, rtol=0)
    assert np.allclose(b, c, atol=1e-6, rtol=0)


def test_weights_frozen(chemeleon):
    from aspect.transform.chemeleon import _load_chemeleon
    mp, _ = _load_chemeleon("cpu")
    assert not mp.training
    assert not any(p.requires_grad for p in mp.parameters())


def test_pipeline_registration():
    from aspect.data import DataPipeline
    pipe = DataPipeline({"emb": ("smiles", "chemeleon")})
    try:
        out = pipe({"smiles": SMILES}).data
    except OSError as e:
        pytest.skip(f"Could not load CheMeleon: {e}")
    assert np.asarray(out["emb"]).shape == (3, 2048)
