"""Missing values: retained columns pass through, transform inputs become NaN."""

from argparse import Namespace

import numpy as np
import pytest
import pyarrow.parquet as pq

from aspect.cli_module import featurize
from aspect.data import DataPipeline

DATA = {
    "x": [1., None, 4., 8.],
    "n": [1, None, 3, 4],
    "y": [0.5, 1.5, None, 2.5],
    "s": ["a", None, "c", "d"],
}


def test_retained_columns_keep_nulls(tmp_path):
    pipe = DataPipeline({"h": ("s", "hash")}, cache_dir=tmp_path)
    out = pipe(DATA, batch_size=2).data
    for column in DATA:
        assert out[column] == DATA[column]


def test_numeric_transform_inputs_propagate_nan(tmp_path):
    pipe = DataPipeline(
        {"lx": ("x", "log"), "n2": ("n", "identity")},
        cache_dir=tmp_path,
    )
    # batch_size=2: one batch of `n` has a null, the other does not.
    out = pipe(DATA, batch_size=2).data
    lx = np.asarray(out["lx"], dtype=float)
    assert np.isnan(lx[1]) and np.all(np.isfinite(lx[[0, 2, 3]]))
    assert np.allclose(lx[[0, 2, 3]], np.log([1., 4., 8.]))
    n2 = np.asarray(out["n2"], dtype=float)
    assert np.isnan(n2[1]) and np.array_equal(n2[[0, 2, 3]], [1., 3., 4.])
    assert out["y"] == DATA["y"]


def _cli_args(tmp_path, **kwargs):
    defaults = dict(
        input_file=str(tmp_path / "data.csv"),
        output=str(tmp_path / "out.parquet"),
        checkpoint=None,
        force_write_data=False,
        features=None,
        extras=None,
        start=None,
        end=None,
        config=None,
        seed=None,
        missing_indicators=False,
        cache=str(tmp_path / "cache"),
    )
    return Namespace(**(defaults | kwargs))


def test_null_free_integers_stay_exact(tmp_path):
    big = [2 ** 53 + 1, 2 ** 53 + 3]
    pipe = DataPipeline({"b2": ("big", "identity")}, cache_dir=tmp_path)
    out = pipe({"big": big}).data
    assert out["b2"] == big


def test_chained_transforms_propagate_nan(tmp_path):
    pipe = DataPipeline({"lx": ("x", ["log", "identity"])}, cache_dir=tmp_path)
    lx = np.asarray(pipe(DATA, batch_size=2).data["lx"], dtype=float)
    assert np.isnan(lx[1]) and np.allclose(lx[[0, 2, 3]], np.log([1., 4., 8.]))


def test_drop_unused_columns_keeps_label_nulls(tmp_path):
    pipe = DataPipeline({"lx": ("x", "log")}, cache_dir=tmp_path)
    out = pipe(DATA, drop_unused_columns=True, keep_extra_columns=["y"]).data
    assert sorted(out.column_names) == ["lx", "y"]
    assert out["y"] == DATA["y"]


def test_cli_config_accepts_pipeline_json(tmp_path):
    (tmp_path / "data.csv").write_text("x,y\n1.0,\n10.0,1.0\n")
    pipe_json = tmp_path / "pipeline.json"
    DataPipeline({"lx": ("x", "log")}).to_config(pipe_json)
    featurize._featurize(_cli_args(tmp_path, config=str(pipe_json)))
    table = pq.read_table(tmp_path / "out.parquet")
    assert np.allclose(table["lx"].to_numpy(), np.log([1., 10.]))
    assert table["y"].null_count == 1


def test_cli_config_accepts_checkpoint_dir(tmp_path, monkeypatch):
    (tmp_path / "data.csv").write_text("x,y\n1.0,\n10.0,1.0\n")
    pipe = DataPipeline({"lx": ("x", "log")}, cache_dir=tmp_path / "cache")
    pipe({"x": [1., 2.]})
    pipe.save(tmp_path / "chk")

    # Only the configuration is read, not the checkpoint's stored data.
    from providence import DataResolver
    monkeypatch.setattr(
        DataResolver, "load",
        classmethod(lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("loaded data"))),
    )
    featurize._featurize(_cli_args(tmp_path, config=str(tmp_path / "chk")))
    assert "lx" in pq.read_table(tmp_path / "out.parquet").column_names


def test_validate_checkpoint_uses_cache_and_does_not_featurize(tmp_path, monkeypatch):
    pipe = DataPipeline({"lx": ("x", "log")}, cache_dir=tmp_path / "cache")
    pipe({"x": [1., 2.]})
    pipe.save(tmp_path / "chk")
    seen = []
    original_load = DataPipeline.load.__func__

    def _load(cls, *args, **kwargs):
        seen.append(kwargs)
        return original_load(cls, *args, **kwargs)

    monkeypatch.setattr(DataPipeline, "load", classmethod(_load))
    monkeypatch.setattr(
        DataPipeline, "__call__",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("featurized")),
    )
    featurize._validate_checkpoints(pipe, tmp_path / "chk")
    assert seen == [{"skip_call": True, "cache_dir": pipe.cache_dir}]


def test_equality_compares_full_config():
    spec = {"lx": ("x", "log")}
    assert DataPipeline(spec) == DataPipeline(spec)
    assert DataPipeline(spec) != DataPipeline(spec, output_format="torch")
    assert DataPipeline(spec) != DataPipeline(spec, output_format_opts={"dtype": "float32"})


def test_from_config_checks_checkpoint_schema(tmp_path):
    pipeline = DataPipeline({"lx": ("x", "log")}).to_config()
    for version in ({}, {"schema_version": 1}):  # aspect <= 0.0.6 wrote none
        assert DataPipeline.from_config(version | {"pipeline": pipeline}) == DataPipeline.from_config(pipeline)
    with pytest.raises(ValueError, match="schema is version 2"):
        DataPipeline.from_config({"schema_version": 2, "pipeline": pipeline})


def test_cli_config_accepts_unversioned_checkpoint_dir(tmp_path):
    import json
    (tmp_path / "data.csv").write_text("x,y\n1.0,\n10.0,1.0\n")
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "config.json").write_text(json.dumps({
        "pipeline": DataPipeline({"lx": ("x", "log")}).to_config(),
        "source": None,
        "artefacts": {"source_data": None, "transformed_data": None, "example": None},
    }))
    featurize._featurize(_cli_args(tmp_path, config=str(tmp_path / "old")))
    assert "lx" in pq.read_table(tmp_path / "out.parquet").column_names


def test_missing_indicators_off_by_default(tmp_path):
    out = DataPipeline({"lx": ("x", "log")}, cache_dir=tmp_path)(DATA).data
    assert not any(column.startswith("missing:") for column in out.column_names)


def test_missing_indicators(tmp_path):
    data = DATA | {"z": [float("nan"), 1., 2., 3.]}
    pipe = DataPipeline(
        {"lx": ("x", "log"), "h": ("s", "hash"), "lz": ("z", ["log", "identity"])},
        missing_indicators=True,
        cache_dir=tmp_path,
    )
    out = pipe(data, batch_size=2, drop_unused_columns=True).data
    assert out["missing:lx"] == [False, True, False, False]
    assert out["missing:h"] == [False, True, False, False]
    assert np.any(np.asarray(out["h"][1]))  # hash of "" looks valid; the flag tells
    assert out["missing:lz"] == [True, False, False, False]  # NaN counts as missing


def test_missing_indicators_are_transforms(tmp_path):
    pipe = DataPipeline({"lx": ("x", "log")}, missing_indicators=True)
    config = pipe.to_config()
    assert "missing_indicators" not in config
    assert config["column_transforms"]["missing:lx"][0]["name"] == "missing"
    # Same pipeline written out by hand, or with the sugar applied again.
    explicit = DataPipeline({"lx": ("x", "log"), "missing:lx": ("x", "missing")})
    assert explicit == pipe
    assert DataPipeline.from_config(config | {"missing_indicators": True}) == pipe
    with pytest.raises(ValueError, match="already defined differently"):
        DataPipeline(
            {"lx": ("x", "log"), "missing:lx": ("x", "identity")},
            missing_indicators=True,
        )


def test_missing_transform_values():
    from aspect.transform.functions import missing
    values = [
        1., float("nan"), np.float32("nan"), np.float64("nan"), np.float32(1.),
        None, "", "CCO", [0, 1], [float("nan")], {"a": None},
    ]
    assert missing()({"a": values}, "a").tolist() == [
        False, True, True, True, False,
        True, True, False, False, False, False,  # structured values are present
    ]


def test_nullable_integers_are_floats_in_transforms(tmp_path):
    # Documented behaviour: integers with nulls are read as float64, so values
    # above 2**53 are rounded; null-free integers stay exact.
    big = 2 ** 53 + 1
    pipe = DataPipeline({"n2": ("n", "identity")}, cache_dir=tmp_path)
    from datasets import Dataset, Features, Value
    source = Dataset.from_dict({"n": [big, None]}, features=Features({"n": Value("int64")}))
    out = pipe(source).data
    assert out["n"] == [big, None]  # source column untouched
    assert out["n2"][0] == float(big) == 2 ** 53 and np.isnan(out["n2"][1])


def test_missing_transform_flags_empty_strings(tmp_path):
    pipe = DataPipeline({"ms": ("s", "missing")}, cache_dir=tmp_path)
    out = pipe({"s": ["CCO", None, ""]}).data
    assert out["ms"] == [False, True, True]


def test_missing_indicator_never_overwrites(tmp_path):
    pipe = DataPipeline({"lx": ("x", "log")}, missing_indicators=True, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="missing:lx already in data"):
        pipe({"x": [1., 2.], "missing:lx": [0, 0]})


def test_cli_missing_indicators_flag(tmp_path):
    (tmp_path / "data.csv").write_text("x,y\n1.0,\n,1.0\n")
    featurize._featurize(_cli_args(
        tmp_path, features=["x:log@lx"], missing_indicators=True,
    ))
    table = pq.read_table(tmp_path / "out.parquet")
    assert table["missing:lx"].to_pylist() == [False, True]
