import json

from aspect.data import DataPipeline
from aspect.io import DataSource


DATA = {
    "x": [1., 2., 4.],
    "labels": [0, 1, 0],
}


def make_pipeline(tmp_path):
    pipeline = DataPipeline(
        {"logx": ["x", "log"]},
        columns_to_keep=["labels"],
        cache_dir=tmp_path / "cache",
    )
    pipeline(DATA)
    return pipeline


def test_pipeline_config_roundtrip(tmp_path):
    pipeline = make_pipeline(tmp_path)

    filename = tmp_path / "pipeline.json"
    expected = pipeline.to_config(filename)

    restored = DataPipeline.from_config(
        str(filename),
        cache_dir=tmp_path / "restored-cache",
    )

    assert restored.to_config() == expected


def test_ephemeral_source_is_packaged(tmp_path):
    pipeline = make_pipeline(tmp_path)

    checkpoint = tmp_path / "checkpoint"

    pipeline.save(checkpoint)

    assert (checkpoint / "config.json").exists()
    assert (checkpoint / "data.parquet").exists()
    assert not (checkpoint / "transformed.parquet").exists()

    assert (checkpoint / "example.parquet").exists()

    restored = DataPipeline.load(
        checkpoint,
        cache_dir=tmp_path / "load-cache",
    )

    assert restored.data_in is not None
    assert len(restored.data_in) == 3
    assert restored.data_out is None


def test_retain_processed_columns(tmp_path):
    pipeline = make_pipeline(tmp_path)

    checkpoint = tmp_path / "checkpoint"

    pipeline.save(
        checkpoint,
        save_transformed_columns=["logx"],
    )

    assert (checkpoint / "transformed.parquet").exists()

    restored = DataPipeline.load(
        checkpoint,
        cache_dir=tmp_path / "load-cache",
    )

    assert restored.data_out is not None
    assert restored.data_out.column_names == ["logx"]
    assert len(restored.data_out) == 3


def test_force_no_source_packaging(tmp_path):
    pipeline = make_pipeline(tmp_path)

    checkpoint = tmp_path / "checkpoint"

    pipeline.save(
        checkpoint,
        save_source_data=False,
    )

    assert not (checkpoint / "data.parquet").exists()

    restored = DataPipeline.load(
        checkpoint,
        cache_dir=tmp_path / "load-cache",
    )

    assert restored.data_in is None


def test_missing_retained_column_raises(tmp_path):
    import pytest

    pipeline = make_pipeline(tmp_path)

    with pytest.raises(
        KeyError,
        match="absent",
    ):
        pipeline.save(
            tmp_path / "checkpoint",
            save_transformed_columns=["does_not_exist"],
        )


def test_remote_source_is_reference_only(tmp_path):
    
    pipeline = make_pipeline(tmp_path)
    pipeline.data_source = DataSource(
        uri="hf://datasets/example/data@abc123:train",
        requested_uri="hf://datasets/example/data@main:train",
        revision="abc123",
        requested_revision="main",
    )

    checkpoint = tmp_path / "checkpoint"
    pipeline.save(checkpoint)

    assert not (checkpoint / "data.parquet").exists()

    with open(checkpoint / "config.json") as file:
        config = json.load(file)

    assert config["source"] == {
        "uri": "hf://datasets/example/data@abc123:train",
        "requested_uri": "hf://datasets/example/data@main:train",
        "revision": "abc123",
        "requested_revision": "main",
    }
