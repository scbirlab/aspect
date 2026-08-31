"""Tests for Chemprop collation."""

from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
import pytest
from rdkit import Chem
import torch

from aspect.collate.chemprop import chemprop_collate


def _cached_molgraph(smiles: str) -> dict:
    """Create an HF-like serialized MolGraph."""
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    molgraph = featurizer(
        Chem.MolFromSmiles(smiles)
    )

    return {
        "V": molgraph.V.tolist(),
        "E": molgraph.E.tolist(),
        "edge_index": molgraph.edge_index.tolist(),
        "rev_edge_index": molgraph.rev_edge_index.tolist(),
    }


def test_chemprop_collate_graphs():
    smiles = ["CCO", "CCN", "CCC"]
    values = [
        {
            "bmg": _cached_molgraph(s),
            "V_d": None,
            "X_d": None,
        }
        for s in smiles
    ]
    batch = chemprop_collate(values)

    assert set(batch) == {
        "bmg",
        "V_d",
        "X_d",
    }
    assert isinstance(
        batch["bmg"],
        BatchMolGraph,
    )

    assert batch["V_d"] is None
    assert batch["X_d"] is None

    assert batch["bmg"].V.ndim == 2
    assert batch["bmg"].edge_index.ndim == 2

    assert torch.unique(
        batch["bmg"].batch
    ).numel() == 3


def test_chemprop_collate_X_d():
    values = [
        {
            "bmg": _cached_molgraph("CCO"),
            "V_d": None,
            "X_d": [1.0, 2.0],
        },
        {
            "bmg": _cached_molgraph("CCN"),
            "V_d": None,
            "X_d": [3.0, 4.0],
        },
    ]

    batch = chemprop_collate(values)

    assert batch["X_d"].shape == (2, 2)
    assert torch.equal(
        batch["X_d"],
        torch.tensor([
            [1., 2.],
            [3., 4.],
        ]),
    )


def test_chemprop_collate_rejects_mixed_optional_values():
    values = [
        {
            "bmg": _cached_molgraph("CCO"),
            "V_d": None,
            "X_d": [1.0, 2.0],
        },
        {
            "bmg": _cached_molgraph("CCN"),
            "V_d": None,
            "X_d": None,
        },
    ]

    with pytest.raises(
        ValueError,
        match="mixture of None and non-None",
    ):
        chemprop_collate(values)
