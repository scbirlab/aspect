from aspect.io import AutoDataset, DataSource


def test_mapping_source_is_ephemeral(tmp_path):
    resolved = AutoDataset.load(
        {
            "x": [
                [1., 2.],
                [3., 4.],
            ],
            "y": [1., 2.],
        },
        cache=tmp_path,
    )

    assert isinstance(resolved.source, DataSource)
    assert resolved.source == DataSource()
    assert len(resolved._dataset) == 2


def test_local_file_source_is_absolute(tmp_path):
    import pandas as pd
    filename = (
        tmp_path / "data.parquet"
    )
    pd.DataFrame(
        {
            "x": [1.,2.],
            "y": [3., 4.],
        }
    ).to_parquet(
        filename,
        index=False,
    )

    resolved = AutoDataset.load(
        str(filename),
        cache=tmp_path / "cache",
    )

    assert resolved.source.uri == str(filename.resolve())
    assert resolved.source.revision is None
    assert resolved.source.requested_revision is None


def test_dataset_source_is_ephemeral():
    from datasets import Dataset

    dataset = Dataset.from_dict(
        {"x": [1., 2.]},
    )
    resolved = AutoDataset.load(dataset)

    assert resolved._dataset is dataset
    assert resolved.source == DataSource()


def test_hf_source_records_resolved_revision(monkeypatch):
    import aspect.io as io
    import datasets

    def fake_load_dataset(
        *,
        path,
        name,
        split,
        revision,
        cache_dir
    ):
        from datasets import Dataset

        assert path == "example/data"
        assert name == "default"
        assert split == "train"
        assert revision == expected_revision

        return Dataset.from_dict(
            {"x": [1., 2.]}
        )

    expected_revision = "0123456789abcdef"
    monkeypatch.setattr(
        io,
        "_resolve_hf_revision",
        lambda repo, revision=None: expected_revision,
    )
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        fake_load_dataset,
    )

    dataset, source = (
        io._resolve_hf_hub_dataset(
            "hf://datasets/example/data@main~default:train"
        )
    )

    assert len(dataset) == 2
    assert source == io.DataSource(
        uri="hf://datasets/example/data@0123456789abcdef~default:train",
        requested_uri="hf://datasets/example/data@main~default:train",
        revision=expected_revision,
        requested_revision="main",
    )
