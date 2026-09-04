
import os

from aspect.io import AutoDataset, dataframe_checksum, file_checksum

def test_dataframe_checksum_is_stable():

    from pandas import DataFrame

    data = DataFrame({
        "x": [1, 2, 3],
        "y": ["a", "b", "c"],
    })

    assert dataframe_checksum(data) == dataframe_checksum(data.copy())


def test_dataframe_checksum_changes_with_schema():

    from pandas import DataFrame

    left = DataFrame({
        "x": [1, 2, 3],
    })
    right = DataFrame({
        "z": [1, 2, 3],
    })

    assert dataframe_checksum(left) != dataframe_checksum(right)


def test_local_source_checksum(tmp_path):

    filename = tmp_path / "data.csv"
    filename.write_text("x\n1\n2\n3\n")

    loaded = AutoDataset.load(str(filename))

    assert loaded.source.checksum == file_checksum(filename)
    assert loaded.source.verify() is True


def test_local_source_verification_detects_change(tmp_path):

    filename = tmp_path / "data.csv"
    filename.write_text(
        "x\n1\n2\n3\n"
    )

    loaded = AutoDataset.load(str(filename))
    filename.write_text(
        "x\n1\n2\n4\n"
    )

    assert loaded.source.verify() is False


def test_dataframe_source_has_checksum():

    data = {
        "x": [1, 2, 3],
    }
    loaded = AutoDataset.load(data)

    assert loaded.source.checksum is not None
    assert len(loaded.source.checksum) == 64


def test_dataframe_cache_does_not_nest_dataset_cache(tmp_path):

    AutoDataset.load(
        {
            "x": [1, 2, 3],
        },
        cache=str(tmp_path),
    )

    assert os.environ["HF_HOME"] == str(tmp_path)
    assert os.environ["HF_DATASETS_CACHE"] == str(tmp_path / "datasets")
    assert not (tmp_path / "datasets" / "datasets").exists()
