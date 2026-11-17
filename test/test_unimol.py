"""Offline tests for frozen Uni-Mol and both conformer paths."""

from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("rdkit")

from aspect.transform import ColumnTransform
from aspect.transform import unimol


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def get_repr(self, inputs):
        self.calls.append(deepcopy(inputs))
        if isinstance(inputs, dict):
            values = [float(np.sum(x)) for x in inputs["coordinates"]]
        else:
            values = [float(len(x)) for x in inputs]
        return {"cls_repr": [np.full(512, value) for value in values]}


@pytest.fixture
def encoder(monkeypatch):
    fake = FakeEncoder()
    monkeypatch.setattr(unimol, "_import_unimol", lambda: (None, None))
    monkeypatch.setattr(unimol, "_load_unimol", lambda *args: fake)
    return fake


def test_conformer_mean_and_missing(encoder):
    rows = [
        {"atomic_numbers": [6, 1], "coordinates": [
            [[1, 0, 0], [0, 0, 0]], [[3, 0, 0], [0, 0, 0]],
        ]},
        None,
        {"atomic_numbers": [8], "coordinates": [[[5, 0, 0]]]},
    ]
    x = unimol.UniMolEmbedding(batch_size=2, device="cpu")({"conf": rows}, "conf")
    assert x.shape == (3, 512) and x.dtype == np.float32
    assert np.array_equal(x[:, 0], [2, 0, 5])
    assert len(encoder.calls) == 2
    assert encoder.calls[0]["atoms"] == [["C", "H"], ["C", "H"]]


def test_empty_conformers_are_zero(encoder):
    row = {"atomic_numbers": [6], "coordinates": np.empty((0, 1, 3), dtype=np.float32)}
    x = unimol.UniMolEmbedding(device="cpu")({"conf": [row]}, "conf")
    assert x.shape == (1, 512)
    assert not x.any()
    assert encoder.calls == []


def test_upstream_list_result(encoder, monkeypatch):
    original = encoder.get_repr

    def as_list(inputs):
        return original(inputs)["cls_repr"]

    monkeypatch.setattr(encoder, "get_repr", as_list)
    x = unimol.UniMolEmbedding(device="cpu")({
        "conf": [{"atomic_numbers": [8], "coordinates": [[[2, 0, 0]]]}]
    }, "conf")
    assert np.array_equal(x[0], np.full(512, 2, dtype=np.float32))


def test_unimol_conformers_from_smiles(encoder):
    f = unimol.UniMolEmbedding(conformers="unimol", batch_size=2, device="cpu")
    x = f({"smiles": ["CCO", None, "not-a-smiles", "O"]}, "smiles")
    assert np.array_equal(x[:, 0], [3, 0, 0, 1])
    assert encoder.calls[0] == ["CCO", "O"]


def test_rejects_wrong_input_and_args(encoder):
    with pytest.raises(TypeError, match="chain `conformers` first"):
        unimol.UniMolEmbedding(device="cpu")({"x": ["CCO"]}, "x")
    for kwargs in ({"batch_size": 0}, {"conformers": "unknown"}, {"device": "mps"}):
        with pytest.raises(ValueError):
            unimol.UniMolEmbedding(**kwargs)


def test_registered_and_serializable(encoder):
    a = ColumnTransform("unimol", "conf")
    b = ColumnTransform("unimol", "conf", kwargs={"remove_hs": True})
    assert a.hash != b.hash
    assert ColumnTransform.from_dict(a.to_dict()).hash == a.hash


def test_model_cache_uses_active_directory(tmp_path, monkeypatch):
    from aspect.cache import hub_cache_dir, use_cache
    seen = []
    monkeypatch.setattr(unimol, "_load_unimol_cached", lambda *args: seen.append(args[-1]))
    for path in (tmp_path / "first", tmp_path / "second"):
        with use_cache(path):
            unimol._load_unimol(2, False, "cpu", "revision")
    assert seen == [hub_cache_dir(tmp_path / "first"), hub_cache_dir(tmp_path / "second")]


def test_model_reused_and_frozen(monkeypatch, tmp_path):
    downloads = []
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo, name, **kwargs: downloads.append(name) or str(tmp_path / name),
    )

    class Parameter:
        requires_grad = True

        def requires_grad_(self, value):
            self.requires_grad = value

    class Model:
        def __init__(self):
            self.parameters_list = [Parameter()]
            self.training = True

        def parameters(self):
            return self.parameters_list

        def eval(self):
            self.training = False

    class FakeUniMol:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.params = {}
            self.model = Model()

    monkeypatch.setattr(unimol, "_import_unimol", lambda: (None, FakeUniMol))
    unimol._load_unimol_cached.cache_clear()
    a = unimol._load_unimol_cached(8, False, "cpu", "revision", str(tmp_path))
    b = unimol._load_unimol_cached(8, False, "cpu", "revision", str(tmp_path))
    assert a is b
    assert downloads == ["mol_pre_all_h_220816.pt", "mol.dict.txt"]
    assert a.kwargs["pretrained_model_path"] == str(tmp_path / downloads[0])
    assert a.params["multi_process"] is False
    assert not a.model.training and not a.model.parameters_list[0].requires_grad
    unimol._load_unimol_cached.cache_clear()
