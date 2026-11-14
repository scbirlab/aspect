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
        data.data,
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
        data.data,
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
        data.data,
        batch_size=2,
    )

    batch = next(iter(loader))

    assert set(batch["molecule"]) == {
        "bmg",
        "V_d",
        "X_d",
    }


def test_dataloader_applies_output_format():
    import torch

    pipeline = DataPipeline({
        "h": ("cat", {"name": "hash", "kwargs": {"ndim": 4}}),
        "y": ("val", "identity"),
    })
    pipeline({"cat": ["a", "b", "c"], "val": [1., 2., 3.]})
    batch = next(iter(pipeline.dataloader(batch_size=3)))
    # float32 Arrow column and float64 Arrow column both arrive as float32
    assert batch["h"].dtype == torch.float32
    assert batch["y"].dtype == torch.float32


def test_dataloader_respects_existing_format():
    import torch

    pipeline = DataPipeline({"y": ("val", "identity")})
    data = pipeline({"val": [1., 2.]})
    loader = pipeline.dataloader(
        data.data.with_format("torch", dtype=torch.float64),
        batch_size=2,
    )
    assert next(iter(loader))["y"].dtype == torch.float64
