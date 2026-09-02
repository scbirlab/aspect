from aspect import DataPipeline

def test_pipeline_dataloader():
    import torch

    pipeline = DataPipeline({"x": ("value", "identity")})

    data = pipeline({
        "value": [
            [1., 2.],
            [3., 4.],
            [5., 6.],
        ],
        "labels": [1., 2., 3.],
    })

    loader = pipeline.dataloader(
        data,
        batch_size=2,
    )

    batch = next(iter(loader))

    assert batch["x"].shape == (2, 2)
    assert torch.is_tensor(batch["x"])


def test_pipeline_dataloader_custom_collator():

    pipeline = DataPipeline({"x": ("value", "identity")})

    data = pipeline({
        "value": [
            [1.],
            [2.],
        ],
    })

    def collate(values):
        return tuple(values)

    loader = pipeline.dataloader(
        data,
        batch_size=2,
        collators={"x": collate},
    )

    batch = next(iter(loader))

    assert isinstance(batch["x"], tuple)


def test_dataloader_uses_discovered_collators():
    pipeline = DataPipeline({
        "molecule": (
            "smiles",
            "chemprop-mol",
        ),
    })

    data = pipeline({
        "smiles": [
            "CCO",
            "c1ccccc1",
        ],
    })

    loader = pipeline.dataloader(
        data,
        batch_size=2,
    )

    batch = next(iter(loader))

    assert set(batch["molecule"]) == {
        "bmg",
        "V_d",
        "X_d",
    }
