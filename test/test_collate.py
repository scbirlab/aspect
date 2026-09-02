
from aspect.collate import ColumnCollator

def test_column_collator_default():
    import torch

    collator = ColumnCollator()

    batch = collator([
        {
            "x": torch.tensor([1., 2.]),
            "y": torch.tensor(1.),
        },
        {
            "x": torch.tensor([3., 4.]),
            "y": torch.tensor(2.),
        },
    ])

    assert batch["x"].shape == (2, 2)
    assert batch["y"].shape == (2,)


def test_column_collator_override():
    collator = ColumnCollator(
        collators={
            "special": lambda values: {
                "values": values,
            }
        }
    )

    batch = collator([
        {
            "x": 1,
            "special": "a",
        },
        {
            "x": 2,
            "special": "b",
        },
    ])

    assert batch["special"] == {
        "values": ["a", "b"],
    }


def test_column_collator_mixed():
    import torch

    collator = ColumnCollator(
        collators={
            "graph": lambda values: {
                "graph_batch": values,
            }
        }
    )

    batch = collator([
        {
            "fp": torch.tensor([1., 2.]),
            "graph": {"id": 1},
            "target": torch.tensor(0.),
        },
        {
            "fp": torch.tensor([3., 4.]),
            "graph": {"id": 2},
            "target": torch.tensor(1.),
        },
    ])

    assert batch["fp"].shape == (2, 2)
    assert batch["target"].shape == (2,)
    assert batch["graph"]["graph_batch"] == [
        {"id": 1},
        {"id": 2},
    ]


def test_pipeline_discovers_registered_collator(monkeypatch):
    from aspect import DataPipeline
    from aspect.transform.registry import (COLLATOR_REGISTRY)

    monkeypatch.setitem(
        COLLATOR_REGISTRY,
        "identity",
        "builtins:list",
    )

    pipeline = DataPipeline({
        "x": (
            "value",
            "identity",
        ),
    })

    assert pipeline.collators["x"] is list


def test_pipeline_collate():
    from aspect import DataPipeline
    pipeline = DataPipeline({
        "x": ("x_raw", "identity"),
    })
    data = pipeline({
        "x_raw": [
            [1., 2.],
            [3., 4.],
        ],
    })
    batch = pipeline.collate(data[:])

    assert batch["x"].shape == (2, 2)


def test_pipeline_collate_chemprop():
    from aspect import DataPipeline
    pipeline = DataPipeline({
        "molecule": ("smiles", "chemprop-mol"),
    })
    data = pipeline({
        "smiles": [
            "CCO",
            "c1ccccc1",
        ],
    })
    batch = pipeline.collate(data[:])

    assert set(batch["molecule"]) == {
        "bmg",
        "V_d",
        "X_d",
    }
