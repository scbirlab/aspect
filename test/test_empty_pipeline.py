from aspect import DataPipeline

def test_empty_pipeline_is_identity():

    data = {
        "x": [
            [0.],
            [1.],
        ],
        "label": [0., 1.],
    }
    pipeline = DataPipeline()
    observed = pipeline(data)

    assert observed.column_names == ["x", "label"]
    assert observed["x"] == data["x"]
    assert observed["label"] == data["label"]
    assert pipeline.data_out is not pipeline.data_in


def test_empty_pipeline_does_not_drop_columns():

    pipeline = DataPipeline()
    data = {
        "x": [1., 2.],
        "y": [3., 4.],
    }
    observed = pipeline(
        data,
        drop_unused_columns=True,
    )

    assert set(observed.column_names) == set(data)
