"""Conformers and frozen Atom-JEPA embeddings.

Offline tests use schemist conformers and a fake encoder. Integration tests
download the pinned weights, and skip only if the Hub cannot be reached, so a
wrong revision or path fails rather than skips.
"""

from argparse import Namespace

import numpy as np
import pyarrow.parquet as pq
import pytest

pytest.importorskip("schemist.conformers")
torch = pytest.importorskip("torch")

from aspect.data import DataPipeline

SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]


def _conformers(smiles, **kwargs):
    from aspect.transform.conformers import Conformers
    return Conformers(**kwargs)({"smiles": smiles}, "smiles")


# conformers ------------------------------------------------------------------


def test_conformers_structure():
    rows = _conformers(SMILES[:1] + [None, "not-a-smiles", "[Fe]"], n_conformers=3)
    z, xyz = rows[0]["atomic_numbers"], rows[0]["coordinates"]
    assert z.dtype == np.int8 and xyz.dtype == np.float32
    assert xyz.shape == (3, len(z), 3)
    assert rows[1:] == [None, None, None]  # missing, invalid, no MMFF94 parameters


def test_conformers_warning_separates_failures(capsys):
    _conformers([None, "not-a-smiles", "[Fe]"])
    assert "2 missing or invalid SMILES, and 1 valid SMILES" in capsys.readouterr().err


def test_partial_embedding_is_null_without_pruning(monkeypatch):
    import schemist.conformers
    original = schemist.conformers.embed_conformers

    def partial(smiles, **kwargs):  # as if only one conformer embedded
        z, xyz = original(smiles, **(kwargs | {"n_conformers": 1}))
        return z, xyz

    monkeypatch.setattr(schemist.conformers, "embed_conformers", partial)
    assert _conformers(SMILES[:1], n_conformers=3) == [None]
    assert _conformers(SMILES[:1], n_conformers=3, prune_rms_thresh=0.5)[0] is not None


def test_conformers_match_schemist():
    from schemist.conformers import embed_conformers
    row = _conformers(SMILES[:1], n_conformers=2, seed=7)[0]
    z, xyz = embed_conformers(SMILES[0], n_conformers=2, seed=7)
    assert np.array_equal(row["atomic_numbers"], z) and np.array_equal(row["coordinates"], xyz)


@pytest.mark.parametrize("kwargs", [
    {"n_conformers": 0}, {"etkdg_version": 5}, {"seed": -1},
    {"n_threads": -1}, {"prune_rms_thresh": float("nan")}, {"max_iters": 0},
])
def test_conformers_invalid_arguments_raise_at_construction(kwargs):
    # Not per row, where they would be mistaken for invalid molecules.
    from aspect.transform.conformers import Conformers
    with pytest.raises(ValueError):
        Conformers(**kwargs)


def test_conformers_saved_to_parquet(tmp_path):
    from aspect.cli_module import featurize
    from schemist.conformers import embed_conformers
    (tmp_path / "data.csv").write_text("smiles\n" + "\n".join(SMILES[:2]) + "\nnot-a-smiles\n")
    featurize._featurize(Namespace(
        input_file=str(tmp_path / "data.csv"), output=str(tmp_path / "out.parquet"),
        checkpoint=None, force_write_data=False, extras=None, start=None, end=None,
        config=None, seed=None, missing_indicators=True, cache=str(tmp_path / "cache"),
        features=["smiles:conformers(n_conformers=2)@conf", "smiles:conformers(n_conformers=2):missing@failed"],
    ))
    table = pq.read_table(tmp_path / "out.parquet")
    assert str(table.schema.field("conf").type) == (
        "struct<atomic_numbers: list<element: int8>, "
        "coordinates: list<element: list<element: list<element: float>>>>"
    )
    rows = table["conf"].to_pylist()
    assert rows[2] is None
    assert table["failed"].to_pylist() == [False, False, True]  # conformers missing
    assert table["missing:conf"].to_pylist() == [False, False, False]  # SMILES present
    expected = embed_conformers(SMILES[1], n_conformers=2)[1]
    assert np.array_equal(np.asarray(rows[1]["coordinates"], dtype=np.float32), expected)


# atom-jepa, with a fake encoder --------------------------------------------


class FakeEncoder:
    """Embeds each conformer as its coordinate sum, repeated; records batches."""

    def __init__(self):
        self.batches = []

    def embed(self, samples, layers="last", batch_size=32):
        self.batches.append(len(samples))
        sums = torch.stack([s["node_coordinates"].sum() for s in samples])
        width = (8, 256) if layers == "all" else (256,)
        return sums.reshape(-1, *([1] * len(width))).expand(-1, *width).clone()


@pytest.fixture
def fake(monkeypatch):
    import aspect.transform.atomjepa as atomjepa
    encoder = FakeEncoder()
    monkeypatch.setattr(atomjepa, "_import_atom_jepa", lambda: (torch, None))
    monkeypatch.setattr(atomjepa, "_load_atom_jepa", lambda *args, **kwargs: encoder)
    return encoder


def _atom_jepa(rows, **kwargs):
    from aspect.transform.atomjepa import AtomJEPAEmbedding
    return AtomJEPAEmbedding(device="cpu", **kwargs)({"conf": rows}, "conf")


def test_averages_conformers_and_zeros_nulls(fake):
    rows = _conformers(SMILES[:2] + ["not-a-smiles"], n_conformers=3)
    x = _atom_jepa(rows)
    assert x.shape == (3, 256) and x.dtype == np.float32
    expected = rows[0]["coordinates"].reshape(3, -1).sum(1).mean()
    assert np.allclose(x[0], expected, rtol=1e-6)
    assert np.all(x[2] == 0)


def test_embeds_in_bounded_chunks(fake):
    rows = _conformers(SMILES, n_conformers=3)  # 9 conformers
    x = _atom_jepa(rows, batch_size=4)
    assert fake.batches == [4, 4, 1]  # molecules may span forward passes
    assert np.allclose(x, _atom_jepa(rows, batch_size=32), rtol=1e-6)


def test_all_layers_flattened(fake):
    assert _atom_jepa(_conformers(SMILES[:1]), layers="all").shape == (1, 2048)


def test_rejects_smiles_input(fake):
    with pytest.raises(TypeError, match="chain `conformers` first"):
        _atom_jepa(SMILES)


def test_shared_conformers_run_once(fake, tmp_path, monkeypatch):
    import schemist.conformers
    calls = []
    original = schemist.conformers.embed_conformers

    def counting(smiles, **kwargs):
        calls.append(smiles)
        return original(smiles, **kwargs)

    monkeypatch.setattr(schemist.conformers, "embed_conformers", counting)
    pipe = DataPipeline({
        "bio": ("smiles", ["conformers", {"name": "atom-jepa", "kwargs": {"device": "cpu"}}]),
        "mol": ("smiles", ["conformers", {"name": "atom-jepa", "kwargs": {"device": "cpu", "model": "molecules"}}]),
    }, cache_dir=tmp_path)
    pipe({"smiles": SMILES})
    assert calls == SMILES


@pytest.mark.parametrize("kwargs", [
    {"model": "crystals"},
    {"layers": 9},
    {"batch_size": 0},
])
def test_invalid_arguments(fake, kwargs):
    from aspect.transform.atomjepa import AtomJEPAEmbedding
    with pytest.raises(ValueError):
        AtomJEPAEmbedding(**kwargs)


def test_revision_is_part_of_cache_key():
    from aspect.transform import ColumnTransform
    a = ColumnTransform("atom-jepa", "smiles")
    b = ColumnTransform("atom-jepa", "smiles", kwargs={"revision": "other"})
    assert a.hash != b.hash


def test_model_cache_follows_active_cache(tmp_path, monkeypatch):
    import aspect.transform.atomjepa as atomjepa
    from aspect.cache import hub_cache_dir, use_cache
    seen = []
    monkeypatch.setattr(atomjepa, "_load_atom_jepa_cached", lambda *args: seen.append(args[-1]))
    for d in (tmp_path / "a", tmp_path / "b"):
        with use_cache(d):
            atomjepa._load_atom_jepa("bio", "rev", "cpu", "fp32")
    assert seen == [hub_cache_dir(tmp_path / "a"), hub_cache_dir(tmp_path / "b")]


# atom-jepa, real weights -----------------------------------------------------


def _load_or_skip(model="bio"):
    pytest.importorskip("atom_jepa")
    from huggingface_hub.errors import LocalEntryNotFoundError
    from aspect.transform.atomjepa import ATOM_JEPA_REVISION, _load_atom_jepa
    try:
        return _load_atom_jepa(model, ATOM_JEPA_REVISION, "cpu", "fp32")
    except LocalEntryNotFoundError as e:  # offline, not cached
        pytest.skip(f"Hugging Face Hub unreachable: {e}")


@pytest.fixture(scope="module")
def encoder():
    return _load_or_skip()


def _pipeline_embed(spec, smiles, tmp_path):
    return np.asarray(DataPipeline({"x": ("smiles", spec)}, cache_dir=tmp_path)({"smiles": smiles}).data["x"])


def test_pinned_weights_architecture(encoder):
    assert (encoder.embedding_dim, encoder.num_layers) == (256, 8)
    assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())


@pytest.mark.parametrize("batch_size", [32, 2])
def test_matches_upstream(encoder, tmp_path, batch_size):
    spec = ["conformers", {"name": "atom-jepa", "kwargs": {"device": "cpu", "batch_size": batch_size}}]
    x = _pipeline_embed(spec, SMILES, tmp_path)
    assert np.array_equal(x, encoder.embed(SMILES, batch_size=batch_size).numpy())


def test_embeds_conformers_read_from_parquet(encoder, tmp_path):
    out = DataPipeline({"conf": ("smiles", "conformers")}, cache_dir=tmp_path)({"smiles": SMILES}).data
    out.to_parquet(str(tmp_path / "conf.parquet"))
    rows = pq.read_table(tmp_path / "conf.parquet")["conf"].to_pylist()
    from aspect.transform.atomjepa import AtomJEPAEmbedding
    x = AtomJEPAEmbedding(device="cpu")({"conf": rows}, "conf")
    assert np.array_equal(x, encoder.embed(SMILES).numpy())


def test_batches_spanning_molecules_match_upstream(encoder, tmp_path):
    # 3 conformers per molecule, 2 per forward pass: molecules span passes.
    from atom_jepa.conformers import smiles_to_samples
    spec = [{"name": "conformers", "kwargs": {"n_conformers": 3}},
            {"name": "atom-jepa", "kwargs": {"device": "cpu", "batch_size": 2}}]
    x = _pipeline_embed(spec, SMILES, tmp_path)
    samples = [c for s in SMILES for c in smiles_to_samples(s, num_conformers=3, seed=42)]
    expected = encoder.embed(samples, batch_size=2).numpy().reshape(len(SMILES), 3, -1).mean(1)
    assert np.allclose(x, expected, atol=1e-6, rtol=0)


def test_conformer_average_matches_upstream(encoder, tmp_path):
    from atom_jepa.conformers import smiles_to_samples
    spec = [{"name": "conformers", "kwargs": {"n_conformers": 3, "seed": 7}},
            {"name": "atom-jepa", "kwargs": {"device": "cpu"}}]
    x = _pipeline_embed(spec, SMILES[:1], tmp_path)
    expected = encoder.embed(smiles_to_samples(SMILES[0], num_conformers=3, seed=7)).numpy().mean(0)
    assert np.allclose(x[0], expected, atol=1e-6, rtol=0)


def test_models_differ(encoder, tmp_path):
    _load_or_skip("molecules")
    spec = lambda m: ["conformers", {"name": "atom-jepa", "kwargs": {"device": "cpu", "model": m}}]
    assert not np.allclose(_pipeline_embed(spec("molecules"), SMILES, tmp_path),
                           _pipeline_embed(spec("bio"), SMILES, tmp_path))


def test_chained_clean_smiles(encoder, tmp_path):
    spec = ["clean-smiles", "conformers", {"name": "atom-jepa", "kwargs": {"device": "cpu"}}]
    a, b = _pipeline_embed(spec, ["CC(=O)O.[Na+]", "CC(=O)[O-]"], tmp_path)
    assert np.array_equal(a, b)
