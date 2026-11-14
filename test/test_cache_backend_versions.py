"""Feature cache must be invalidated when backend package versions change."""

import aspect.data
from aspect.data import DataPipeline


def _featurize_fingerprint(csv_path, cache_dir) -> str:
    pipe = DataPipeline(
        {"vector": ("category", {"name": "hash", "kwargs": {"ndim": 8}})},
        cache_dir=str(cache_dir),
    )
    out = pipe(str(csv_path), cache_dir=str(cache_dir))
    return out.data._fingerprint


def test_backend_version_change_invalidates_cache(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("category,x\na,1.0\nb,2.0\nc,3.0\n")
    cache_dir = tmp_path / "cache"

    # raising=False so that, without the fix, this fails on the
    # fingerprint assertion rather than on a missing attribute.
    monkeypatch.setattr(
        aspect.data, "_backend_versions", lambda: {"schemist": "0.0.7"},
        raising=False,
    )
    first = _featurize_fingerprint(csv_path, cache_dir)
    repeat = _featurize_fingerprint(csv_path, cache_dir)

    monkeypatch.setattr(
        aspect.data, "_backend_versions", lambda: {"schemist": "0.0.8"},
        raising=False,
    )
    upgraded = _featurize_fingerprint(csv_path, cache_dir)

    assert first == repeat
    assert upgraded != first


def test_featurize_accepts_backend_versions():
    out = DataPipeline._featurize(
        {"category": ["a", "b"]},
        column_transforms={
            "vector": (
                {
                    "name": "hash",
                    "input_column": "category",
                    "kwargs": {"ndim": 8},
                },
            ),
        },
        backend_versions={"schemist": "0.1.0"},
    )
    assert len(out["vector"]) == 2
