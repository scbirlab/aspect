

from aspect.data import DataPipeline
from aspect.io import AutoDataset

import numpy as np

def test_mapping_preserves_array_columns():
    data = {
        "x": [
            [1., 2.],
            [3., 4.],
        ],
        "y": np.asarray([1., 2.]),
    }

    dataset = AutoDataset.load(data)._dataset

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
        prepared["logx"],
        np.log([
            [1., 2.],
            [3., 4.],
        ]),
    )
