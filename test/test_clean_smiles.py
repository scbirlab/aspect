"""clean-smiles as an explicit, shareable pipeline step."""

from argparse import Namespace

import pyarrow.parquet as pq
import pytest

pytest.importorskip("schemist")

from aspect.data import DataPipeline

SMILES = ["CC(=O)O.[Na+]", "CCO", "nonsense", None]


def test_clean_smiles_transform(tmp_path):
    out = DataPipeline({"clean": ("smiles", "clean-smiles")}, cache_dir=tmp_path)(
        {"smiles": SMILES}
    ).data
    assert out["clean"] == ["CC(=O)[O-]", "CCO", None, None]
    assert out["smiles"] == SMILES


def test_clean_smiles_options(tmp_path):
    spec = ("smiles", {"name": "clean-smiles", "kwargs": {"protonate": False}})
    out = DataPipeline({"clean": spec}, cache_dir=tmp_path)({"smiles": SMILES[:1]}).data
    assert out["clean"] == ["CC(=O)O"]


def test_shared_cleaning_runs_once(tmp_path, monkeypatch, capsys):
    import aspect.transform.cleaning as cleaning
    calls = []
    original = cleaning.clean_smiles

    def counting(smiles, **kwargs):
        calls.append(kwargs)
        return original(smiles, **kwargs)

    monkeypatch.setattr(cleaning, "clean_smiles", counting)
    pipe = DataPipeline({
        "clean": ("smiles", "clean-smiles"),
        "fp": ("smiles", ["clean-smiles", "morgan-fingerprint"]),
        "h": ("smiles", ["clean-smiles", "hash"]),
    }, cache_dir=tmp_path)
    out = pipe({"smiles": SMILES}, batch_size=len(SMILES)).data
    assert len(calls) == 1
    assert sorted(out.column_names) == ["clean", "fp", "h", "smiles"]
    assert "already present" not in capsys.readouterr().err


def test_different_cleaning_is_not_shared(tmp_path):
    pipe = DataPipeline({
        "a": ("smiles", "clean-smiles"),
        "b": ("smiles", {"name": "clean-smiles", "kwargs": {"protonate": False}}),
    }, cache_dir=tmp_path)
    out = pipe({"smiles": SMILES[:1]}).data
    assert out["a"] == ["CC(=O)[O-]"] and out["b"] == ["CC(=O)O"]


def test_cli_chain(tmp_path):
    from aspect.cli_module import featurize
    (tmp_path / "data.csv").write_text("smiles\nCC(=O)O.[Na+]\nCCO\n")
    featurize._featurize(Namespace(
        input_file=str(tmp_path / "data.csv"),
        output=str(tmp_path / "out.parquet"),
        checkpoint=None, force_write_data=False, extras=None, start=None,
        end=None, config=None, seed=None, missing_indicators=False,
        cache=str(tmp_path / "cache"),
        features=["smiles:clean-smiles@clean", "smiles:clean-smiles:hash@h"],
    ))
    table = pq.read_table(tmp_path / "out.parquet")
    assert table["clean"].to_pylist() == ["CC(=O)[O-]", "CCO"]
    assert "h" in table.column_names


def test_shared_step_survives_save_and_reload(tmp_path, monkeypatch):
    import numpy as np
    import aspect.transform.cleaning as cleaning
    pipe = DataPipeline({
        "h64": ("smiles", ["clean-smiles", {"name": "hash", "kwargs": {"ndim": 64}}]),
        "h128": ("smiles", ["clean-smiles", {"name": "hash", "kwargs": {"ndim": 128}}]),
    }, cache_dir=tmp_path / "cache")
    before = pipe({"smiles": SMILES}).data
    pipe.save(tmp_path / "chk")
    restored = DataPipeline.load(tmp_path / "chk", skip_call=True, cache_dir=tmp_path / "cache2")
    assert restored == pipe

    calls = []
    original = cleaning.clean_smiles

    def counting(smiles, **kwargs):
        calls.append(1)
        return original(smiles, **kwargs)

    monkeypatch.setattr(cleaning, "clean_smiles", counting)
    after = restored({"smiles": SMILES}, batch_size=len(SMILES)).data
    assert len(calls) == 1  # still one shared cleaning step
    for column in ("h64", "h128"):
        assert np.array_equal(np.asarray(after[column]), np.asarray(before[column]))
