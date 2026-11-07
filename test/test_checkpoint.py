import json

import pandas as pd
import pytest

from aspect.data import DataPipeline
from providence import DataSource


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
    assert (checkpoint / "data.pv" / "data.parquet").exists()
    assert not (checkpoint / "transformed.pv" / "data.parquet").exists()

    assert (checkpoint / "data.pv" / "example.parquet").exists()

    restored = DataPipeline.load(
        checkpoint,
        cache_dir=tmp_path / "load-cache",
    )

    assert restored.data_in is not None
    assert len(restored.data_in.data) == 3
    assert restored.data_out is not None

    restored = DataPipeline.load(
        checkpoint,
        cache_dir=tmp_path / "load-cache",
        skip_call=True,
    )

    assert restored.data_in is not None
    assert len(restored.data_in.data) == 3
    assert restored.data_out is None


def test_force_no_source_packaging(tmp_path):
    pipeline = make_pipeline(tmp_path)

    assert pipeline.data_in.data_source.uri is None

    checkpoint = tmp_path / "checkpoint"

    with pytest.raises(
        ValueError,
        match="Reference-only checkpoint requires a source URI",
    ):
        pipeline.save(
            checkpoint,
            save_source_data=False,
        )

    # Automatic policy should instead materialise ephemeral data.
    pipeline.save(checkpoint)

    assert (checkpoint / "data.pv" / "data.parquet").exists()

    restored = DataPipeline.load(checkpoint)
    assert restored.data_in.data["x"] == DATA["x"]


def test_remote_source_is_reference_only(tmp_path):
    pipeline = make_pipeline(tmp_path)

    source = DataSource(
        uri="hf://datasets/example/data@abc123:train",
        requested_uri="hf://datasets/example/data@main:train",
        revision="abc123",
        requested_revision="main",
    )

    # Provenance belongs to the input resolver, not the pipeline.
    pipeline.data_in.data_source = source

    checkpoint = tmp_path / "checkpoint"
    pipeline.save(checkpoint)

    source_checkpoint = checkpoint / "data.pv"

    assert (source_checkpoint / "config.json").exists()
    assert not (source_checkpoint / "data.parquet").exists()

    with open(source_checkpoint / "config.json") as f:
        config = json.load(f)

    assert config["source"] == source.to_config()
    assert config["artefacts"]["source_data"] is None
    assert config["artefacts"]["data_checksum"] is None
    assert config["source"]["revision"] == "abc123"


def test_retain_processed_columns(tmp_path):
    pipeline = make_pipeline(tmp_path)

    checkpoint = tmp_path / "checkpoint"

    pipeline.save(
        checkpoint,
        save_transformed=True,
    )

    assert (checkpoint / "transformed.pv").exists()

    restored = DataPipeline.load(
        checkpoint,
        cache_dir=tmp_path / "load-cache",
    )

    assert restored.data_out is not None
    assert restored.data_out.data.column_names == pipeline.data_out.data.column_names
    assert len(restored.data_out.data) == 3


def test_checkpoint_reconstructs_verified_local_source(tmp_path):

    source = tmp_path / "train.parquet"

    pd.DataFrame({
        "x": [1., 2., 3.],
        "y": [2., 4., 6.],
    }).to_parquet(
        source,
        index=False,
    )

    pipeline = DataPipeline()
    pipeline(str(source))

    checkpoint = tmp_path / "pipeline"

    pipeline.save(
        checkpoint,
        save_source_data=False,
    )

    restored = DataPipeline.load(checkpoint)

    assert restored.data_in is not None
    assert restored.data_in.data_source.verify() is True


def test_checkpoint_rejects_modified_local_source(tmp_path):

    source = tmp_path / "train.parquet"

    pd.DataFrame({
        "x": [1., 2., 3.],
    }).to_parquet(
        source,
        index=False,
    )

    pipeline = DataPipeline()
    pipeline(str(source))

    checkpoint = tmp_path / "pipeline"

    pipeline.save(
        checkpoint,
        save_source_data=False,
    )

    pd.DataFrame({
        "x": [1., 2., 4.],
    }).to_parquet(
        source,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="checksum",
    ):
        DataPipeline.load(checkpoint)


def test_embedded_checkpoint_ignores_modified_original_source(
    tmp_path,
):

    source = tmp_path / "train.parquet"

    pd.DataFrame({
        "x": [1., 2., 3.],
    }).to_parquet(
        source,
        index=False,
    )

    pipeline = DataPipeline()
    pipeline(str(source))

    checkpoint = tmp_path / "pipeline"

    pipeline.save(
        checkpoint,
        save_source_data=True,
    )

    pd.DataFrame({
        "x": [9., 9., 9.],
    }).to_parquet(
        source,
        index=False,
    )

    restored = DataPipeline.load(checkpoint)

    assert restored.data_in.data["x"] == [1., 2., 3.]
