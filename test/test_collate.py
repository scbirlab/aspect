
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
