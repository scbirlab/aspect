"""Multi-conformer 3D descriptors."""

import numpy as np
import pytest

pytest.importorskip("schemist")

from aspect.data import DataPipeline
from aspect.transform import ColumnTransform


SMILES = ["c1ccccc1", "CCCCCCCCCCO", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "not-a-smiles"]


def _featurize(**kwargs):
    pipe = DataPipeline(
        {"d3": ("smiles", {"name": "descriptors-3d", "kwargs": kwargs})}
    )
    return np.asarray(pipe({"smiles": SMILES}).data["d3"])


def test_width_independent_of_molecule():
    # rigid (pruned to 1 conformer), flexible, invalid: same width
    x = _featurize()
    assert x.shape == (len(SMILES), 22)
    assert np.all(x[-1] == 0.)
    # benzene: conformers collapse on pruning, so std columns are 0
    assert np.allclose(x[0, 11:], 0.)


def test_explicit_single_conformer_with_std_keeps_width():
    assert _featurize(n_conformers=1, with_std=True).shape == (len(SMILES), 22)
    assert _featurize(n_conformers=1).shape == (len(SMILES), 11)


def test_deterministic_and_seeded():
    a, b = _featurize(seed=1), _featurize(seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, _featurize(seed=2))


def test_options_change_output_column():
    cols = {
        ColumnTransform(
            name="descriptors-3d", input_column="smiles", kwargs=kw
        ).output_column
        for kw in ({}, {"n_conformers": 5}, {"max_conformers": 20})
    }
    assert len(cols) == 3
