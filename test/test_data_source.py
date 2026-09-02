from aspect.data import DataPipeline
from aspect.io import DataSource


def test_pipeline_retains_data_source(tmp_path):
    pipeline = DataPipeline(
        {"logx": ["x", "log"]},
        cache_dir=tmp_path,
    )

    pipeline({
        "x": [
            [1., 2.],
            [3., 4.],
        ],
    })

    assert isinstance(pipeline.data_source, DataSource)
    assert pipeline.data_source == DataSource()
