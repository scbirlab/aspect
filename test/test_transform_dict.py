

from aspect.data import DataPipeline
from providence import DataResolver
import numpy as np

def test_mapping_preserves_array_columns():
    data = {
        "x": [
            [1., 2.],
            [3., 4.],
        ],
        "y": np.asarray([1., 2.]),
    }

    dataset = DataResolver(data).data

    assert np.allclose(
        dataset["x"],
        [
            [1., 2.],
            [3., 4.],
        ],
    )


def test_pipeline_transforms_array_column():
    pipeline = DataPipeline(
        column_transforms={
            "logx": ("x", "log"),
        }
    )

    prepared = pipeline({
        "x": [
            [1., 2.],
            [3., 4.],
        ]
    })

    assert np.allclose(
        prepared.data["logx"],
        np.log([
            [1., 2.],
            [3., 4.],
        ]),
    )
