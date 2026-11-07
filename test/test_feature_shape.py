from datasets import Dataset
from aspect.data import DataPipeline

def test_scalar_and_vector_features():
    ds = Dataset.from_dict({
        "x": [1.0, 2.0, 3.0],
        "category": ["a", "b", "c"],
    })

    pipe = DataPipeline({
        "scalar": ("x", "identity"),
        "vector": (
            "category",
            {"name": "hash", "kwargs": {"ndim": 8}},
        ),
    })

    out = pipe(ds).data

    assert isinstance(out[0]["scalar"], (int, float))
    assert len(out[0]["vector"]) == 8