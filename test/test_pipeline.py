"""Tests for the featurization pipeline."""

import json
import tempfile
import os

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

from aspect.serializing import Preprocessor
from aspect.pipeline.data import (
    ColumnPipeline,
    DataPipeline,
    X_KEY,
    Y_KEY,
    DEFAULT_BATCH_SIZE,
)
from aspect.pipeline.io import AutoDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0.1, 0.2, 0.3, 0.4]})


@pytest.fixture
def multi_col_df():
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [4.0, 5.0, 6.0],
        "label": [0.0, 1.0, 0.0],
    })


@pytest.fixture
def string_df():
    return pd.DataFrame({
        "category": ["foo", "bar", "foo", "baz"],
        "score": [1.0, 2.0, 3.0, 4.0],
    })


# ---------------------------------------------------------------------------
# Preprocessor tests
# ---------------------------------------------------------------------------

class TestPreprocessor:

    def test_identity_call(self):
        p = Preprocessor(name="identity", input_column="x")
        out = p({"x": [1.0, 2.0, 3.0]})
        assert p.output_column in out
        np.testing.assert_allclose(out[p.output_column], [1.0, 2.0, 3.0])

    def test_log_call(self):
        p = Preprocessor(name="log", input_column="v")
        vals = [1.0, np.e, np.e ** 2]
        out = p({"v": vals})
        np.testing.assert_allclose(out[p.output_column], [0.0, 1.0, 2.0])

    def test_onehot_call(self):
        p = Preprocessor(name="one-hot", input_column="cat",
                         kwargs={"categories": ["a", "b", "c"]})
        out = p({"cat": ["a", "c", "b"]})
        expected = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
        assert out[p.output_column].tolist() == expected

    def test_hash_call(self):
        p = Preprocessor(name="hash", input_column="tag", kwargs={"ndim": 16, "seed": 0})
        out = p({"tag": ["foo", "foo", "bar"]})
        arr = out[p.output_column]
        assert arr.shape == (3, 16)
        np.testing.assert_array_equal(arr[0], arr[1])
        assert not np.array_equal(arr[0], arr[2])

    def test_show_returns_registered_names(self):
        names = Preprocessor.show()
        assert "identity" in names
        assert "log" in names
        assert "one-hot" in names
        assert "hash" in names

    def test_to_dict_round_trip(self):
        p = Preprocessor(name="hash", input_column="x", kwargs={"ndim": 8, "seed": 1})
        d = p.to_dict()
        assert d == {"name": "hash", "input_column": "x", "kwargs": {"ndim": 8, "seed": 1}}
        p2 = Preprocessor.from_dict(d)
        assert p2.name == p.name
        assert p2.input_column == p.input_column
        assert p2.output_column == p.output_column

    def test_to_file_round_trip(self):
        p = Preprocessor(name="identity", input_column="col")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            p.to_file(fname)
            with open(fname) as fh:
                loaded = json.load(fh)
            assert loaded["name"] == "identity"
            p2 = Preprocessor.from_file(fname)
            assert p2.output_column == p.output_column
        finally:
            os.unlink(fname)

    def test_output_column_is_deterministic(self):
        p1 = Preprocessor(name="identity", input_column="x")
        p2 = Preprocessor(name="identity", input_column="x")
        assert p1.output_column == p2.output_column

    def test_different_kwargs_give_different_output_column(self):
        p1 = Preprocessor(name="hash", input_column="x", kwargs={"ndim": 8})
        p2 = Preprocessor(name="hash", input_column="x", kwargs={"ndim": 16})
        assert p1.output_column != p2.output_column

    def test_unknown_function_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            Preprocessor(name="no-such-function", input_column="x")


# ---------------------------------------------------------------------------
# ColumnPipeline tests
# ---------------------------------------------------------------------------

class TestColumnPipeline:

    def test_no_steps_output_is_input_column(self):
        cp = ColumnPipeline("raw", [])
        assert cp.output_column == "raw"

    def test_single_step_output_column(self):
        cp = ColumnPipeline("score", [Preprocessor("log", "score")])
        assert cp.output_column.startswith("aspect/")
        assert "log" in cp.output_column

    def test_chaining_wires_input_columns(self):
        p0 = Preprocessor("identity", "x")
        p1 = Preprocessor("log", "x")
        cp = ColumnPipeline("x", [p0, p1])
        assert cp.steps[0].input_column == "x"
        assert cp.steps[1].input_column == cp.steps[0].output_column

    def test_apply_adds_output_column(self):
        ds = Dataset.from_dict({"x": [1.0, 4.0, 9.0]})
        cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        out = cp.apply(ds)
        assert cp.output_column in out.column_names

    def test_apply_chained_steps(self):
        ds = Dataset.from_dict({"x": [1.0, np.e, np.e ** 2]})
        cp = ColumnPipeline("x", [
            Preprocessor("identity", "x"),
            Preprocessor("log", "x"),
        ])
        out = cp.apply(ds)
        arr = np.asarray(out.with_format("numpy")[cp.output_column])
        np.testing.assert_allclose(arr, [0.0, 1.0, 2.0], atol=1e-6)

    def test_to_dict_no_input_column_in_steps(self):
        cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        d = cp.to_dict()
        assert d["input_column"] == "x"
        assert "input_column" not in d["steps"][0]

    def test_from_dict_round_trip(self):
        cp = ColumnPipeline("x", [Preprocessor("hash", "x", kwargs={"ndim": 32})])
        cp2 = ColumnPipeline.from_dict(cp.to_dict())
        assert cp2.input_column == "x"
        assert cp2.output_column == cp.output_column

    def test_to_file_from_file(self):
        cp = ColumnPipeline("tag", [Preprocessor("hash", "tag", kwargs={"ndim": 8})])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            cp.to_file(fname)
            cp2 = ColumnPipeline.from_file(fname)
            assert cp2.input_column == cp.input_column
            assert cp2.output_column == cp.output_column
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# DataPipeline tests
# ---------------------------------------------------------------------------

class TestDataPipeline:

    def test_single_group_fit_transform(self, simple_df):
        cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        ds = pipe.fit_transform(simple_df)
        assert X_KEY in ds.column_names
        assert Y_KEY in ds.column_names
        assert pipe.input_shape == (1,)
        assert pipe.output_shape == (1,)

    def test_single_group_values_correct(self, simple_df):
        cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        ds = pipe.fit_transform(simple_df)
        X = np.asarray(ds.with_format(None)[X_KEY])
        np.testing.assert_allclose(X.ravel(), simple_df["x"].values)

    def test_multi_group_output_keys(self, multi_col_df):
        cp_a = ColumnPipeline("a", [Preprocessor("identity", "a")])
        cp_b = ColumnPipeline("b", [Preprocessor("identity", "b")])
        pipe = DataPipeline(column_groups=[[cp_a], [cp_b]], label_columns=["label"])
        ds = pipe.fit_transform(multi_col_df)
        assert f"{X_KEY}:0000" in ds.column_names
        assert f"{X_KEY}:0001" in ds.column_names
        assert isinstance(pipe.input_shape, tuple)
        assert len(pipe.input_shape) == 2

    def test_same_column_in_multiple_groups(self, multi_col_df):
        cp_a = ColumnPipeline("a", [Preprocessor("identity", "a")])
        cp_b = ColumnPipeline("b", [Preprocessor("identity", "b")])
        # cp_a appears in both groups
        pipe = DataPipeline(column_groups=[[cp_a, cp_b], [cp_a]], label_columns=["label"])
        ds = pipe.fit_transform(multi_col_df)
        assert f"{X_KEY}:0000" in ds.column_names
        assert f"{X_KEY}:0001" in ds.column_names
        raw = ds.with_format(None)
        grp0 = np.asarray(raw[f"{X_KEY}:0000"])
        grp1 = np.asarray(raw[f"{X_KEY}:0001"])
        assert grp0.shape == (3, 2)  # a+b concatenated
        assert grp1.shape == (3, 1)  # just a

    def test_deduplication_applied_once(self, multi_col_df):
        cp_a = ColumnPipeline("a", [Preprocessor("identity", "a")])
        pipe = DataPipeline(column_groups=[[cp_a], [cp_a]], label_columns=[])
        assert len(pipe._unique_column_pipelines()) == 1

    def test_no_labels(self, simple_df):
        cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        pipe = DataPipeline(column_groups=[[cp]], label_columns=[])
        ds = pipe.fit_transform(simple_df)
        assert X_KEY in ds.column_names
        assert Y_KEY not in ds.column_names
        assert pipe.output_shape is None

    def test_transform_separate_from_fit(self, simple_df):
        train = simple_df.iloc[:2]
        test = simple_df.iloc[2:]
        cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        pipe.fit_transform(train)
        ds_test = pipe.transform(test)
        assert X_KEY in ds_test.column_names
        assert len(ds_test) == 2

    def test_multi_label_columns(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y1": [0.1, 0.2], "y2": [0.3, 0.4]})
        cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        pipe = DataPipeline(column_groups=[[cp]], label_columns=["y1", "y2"])
        ds = pipe.fit_transform(df)
        assert pipe.output_shape == (2,)

    def test_string_hashing(self, string_df):
        cp = ColumnPipeline("category", [
            Preprocessor("hash", "category", kwargs={"ndim": 16})
        ])
        pipe = DataPipeline(column_groups=[[cp]], label_columns=["score"])
        ds = pipe.fit_transform(string_df)
        assert pipe.input_shape == (16,)

    def test_chained_steps_pipeline(self, simple_df):
        cp = ColumnPipeline("x", [
            Preprocessor("identity", "x"),
            Preprocessor("log", "x"),
        ])
        pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        ds = pipe.fit_transform(simple_df)
        X = np.asarray(ds.with_format(None)[X_KEY])
        np.testing.assert_allclose(X.ravel(), np.log(simple_df["x"].values), atol=1e-6)

    def test_from_hf_dataset(self, simple_df):
        raw_ds = Dataset.from_pandas(simple_df)
        cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        ds = pipe.fit_transform(raw_ds)
        assert X_KEY in ds.column_names

    # ------ JSON round-trip ------

    def test_to_dict_structure(self, simple_df):
        cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        d = pipe.to_dict()
        assert "column_groups" in d
        assert "label_columns" in d
        assert d["label_columns"] == ["y"]
        assert d["column_groups"][0][0]["input_column"] == "x"

    def test_to_dict_is_json_serializable(self, simple_df):
        cp = ColumnPipeline("x", [Preprocessor("hash", "x", kwargs={"ndim": 8})])
        pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        d = pipe.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_from_dict_produces_same_outputs(self, simple_df):
        cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        ds1 = pipe.fit_transform(simple_df)
        pipe2 = DataPipeline.from_dict(pipe.to_dict())
        ds2 = pipe2.transform(simple_df)
        X1 = np.asarray(ds1.with_format(None)[X_KEY])
        X2 = np.asarray(ds2.with_format(None)[X_KEY])
        np.testing.assert_allclose(X1, X2)

    def test_to_file_from_file(self, simple_df):
        cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            pipe.to_file(fname)
            pipe2 = DataPipeline.from_file(fname)
            assert pipe2.label_columns == ["y"]
            assert pipe2.column_groups[0][0].input_column == "x"
        finally:
            os.unlink(fname)

    def test_multi_group_json_round_trip(self, multi_col_df):
        cp_a = ColumnPipeline("a", [Preprocessor("identity", "a")])
        cp_b = ColumnPipeline("b", [Preprocessor("identity", "b")])
        pipe = DataPipeline(column_groups=[[cp_a, cp_b], [cp_a]], label_columns=["label"])
        pipe2 = DataPipeline.from_dict(pipe.to_dict())
        ds = pipe2.fit_transform(multi_col_df)
        assert f"{X_KEY}:0000" in ds.column_names

    # ------ Checkpoint ------

    def test_checkpoint_creates_files(self, simple_df):
        cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        pipe.fit_transform(simple_df)
        with tempfile.TemporaryDirectory() as d:
            pipe.checkpoint(d)
            assert os.path.exists(os.path.join(d, "pipeline-spec.json"))
            assert os.path.isdir(os.path.join(d, "input-data.hf"))
            assert os.path.isdir(os.path.join(d, "training-data.hf"))

    def test_checkpoint_round_trip(self, simple_df):
        cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        pipe = DataPipeline(column_groups=[[cp]], label_columns=["y"])
        pipe.fit_transform(simple_df)
        with tempfile.TemporaryDirectory() as d:
            pipe.checkpoint(d)
            pipe2 = DataPipeline.load_checkpoint(d)
        assert pipe2.label_columns == ["y"]
        assert pipe2.training_data is not None
        assert pipe2.input_shape == pipe.input_shape
        assert pipe2.output_shape == pipe.output_shape

    def test_checkpoint_without_training_data(self):
        cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        pipe = DataPipeline(column_groups=[[cp]], label_columns=[])
        with tempfile.TemporaryDirectory() as d:
            pipe.checkpoint(d)
            assert os.path.exists(os.path.join(d, "pipeline-spec.json"))
            pipe2 = DataPipeline.load_checkpoint(d)
        assert pipe2.training_data is None

    # ------ AutoDataset ------

    def test_auto_dataset_from_dict(self):
        ds = AutoDataset.load({"x": [1, 2, 3]})
        assert list(ds._dataset["x"]) == [1, 2, 3]

    def test_auto_dataset_from_dataframe(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        ds = AutoDataset.load(df)
        assert sorted(ds._dataset.column_names) == ["a", "b"]

    def test_auto_dataset_passthrough(self):
        raw = Dataset.from_dict({"z": [7, 8, 9]})
        ds = AutoDataset.load(raw)
        assert ds._dataset is raw

    def test_make_dataloader_not_implemented(self):
        cp = ColumnPipeline("x", [Preprocessor("identity", "x")])
        pipe = DataPipeline(column_groups=[[cp]], label_columns=[])
        with pytest.raises(NotImplementedError):
            pipe.make_dataloader(None, 32, True)
