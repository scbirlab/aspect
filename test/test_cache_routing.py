"""Transforms see the pipeline's cache (CLI --cache) via the shared providence cache."""

import os

import numpy as np

from aspect.cache import CACHE_ENV, resolve_cache, use_cache
from aspect.data import DataPipeline
from aspect.transform.registry import FUNCTION_REGISTRY, register_function

SEEN: list[str] = []

if "_test-record-cache" not in FUNCTION_REGISTRY:
    @register_function("_test-record-cache")
    def _record_cache():
        def _fn(data, input_column):
            SEEN.append(resolve_cache())
            return np.zeros((len(data[input_column]), 1), dtype=np.float32)
        return _fn


def test_pipeline_cache_reaches_transforms(tmp_path, monkeypatch):
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "env-cache"))
    explicit = tmp_path / "cli-cache"
    SEEN.clear()
    pipe = DataPipeline({"r": ("x", "_test-record-cache")}, cache_dir=str(explicit))
    pipe({"x": [1.0, 2.0, 3.0]})
    assert SEEN and set(SEEN) == {str(explicit)}
    # restored afterwards
    assert os.environ[CACHE_ENV] == str(tmp_path / "env-cache")


def test_pipeline_default_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "env-cache"))
    assert DataPipeline({}).cache_dir == str(tmp_path / "env-cache")


def test_call_cache_overrides_pipeline_cache(tmp_path):
    SEEN.clear()
    pipe = DataPipeline({"r": ("x", "_test-record-cache")}, cache_dir=str(tmp_path / "a"))
    pipe({"x": [4.0, 5.0]}, cache_dir=str(tmp_path / "b"))
    assert set(SEEN) == {str(tmp_path / "b")}


def test_use_cache_restores_unset(tmp_path, monkeypatch):
    monkeypatch.delenv(CACHE_ENV, raising=False)
    with use_cache(tmp_path) as c:
        assert os.environ[CACHE_ENV] == c
    assert CACHE_ENV not in os.environ


def test_hf_transforms_load_lazily(monkeypatch):
    import pytest
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import aspect.transform.deep_functions as df

    def _boom(*args, **kwargs):
        raise AssertionError("model loaded at construction")

    monkeypatch.setattr(df, "_load_bart_model", _boom)
    monkeypatch.setattr(df, "_load_esm2_cached", _boom)
    df.HfBART(ref="any/model")
    df.ESM2(model="any/model", revision="x")


def test_hf_models_download_into_pipeline_cache(tmp_path):
    import pytest
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    ref = "hf-internal-testing/tiny-random-BartForConditionalGeneration"
    cache = tmp_path / "cli-cache"
    pipe = DataPipeline(
        {"e": ("s", {"name": "hf-bart", "kwargs": {"ref": ref, "clean_smiles": False}})},
        cache_dir=str(cache),
    )
    try:
        pipe({"s": ["CCO", "c1ccccc1"]})
    except OSError as e:
        pytest.skip(f"Could not load {ref}: {e}")
    hub = cache / "hub"
    assert any(p.name.startswith("models--hf-internal-testing") for p in hub.iterdir())
