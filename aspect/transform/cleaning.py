"""SMILES cleaning shared by molecular featurizers."""

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from .registry import register_function

if TYPE_CHECKING:
    from rdkit.Chem import Mol
else:
    Mol = Any

DEFAULT_PH: float = 7.4


def _import_schemist():
    try:
        from schemist.cleaning import clean_mol, sanitize_smiles_to_mol
    except ImportError as e:
        raise ImportError(
            f"SMILES cleaning needs schemist ({e}). Try `pip install aspect-data[chem]`."
        ) from e
    return clean_mol, sanitize_smiles_to_mol


def clean_mols(
    smiles: Iterable[str | None],
    desalt: bool = True,
    protonate: bool = True,
    protonation_pH: float = DEFAULT_PH,
) -> list[Mol | None]:
    """Parse SMILES to RDKit Mols, optionally desalting and setting protonation state.

    Uses schemist: parsing with sanifix and stereo perception, desalting with
    RDKit `FragmentParent`, and protonation with Dimorphite-DL (the single
    dominant state at `protonation_pH`; needs `schemist[protonate]`).
    With `desalt` and `protonate` both off, this is plain `MolFromSmiles`.

    Returns `None` for missing, empty, or unparseable SMILES.

    Examples
    ========
    >>> from rdkit.Chem import MolToSmiles
    >>> def show(*args, **kwargs):
    ...     return [m if m is None else MolToSmiles(m) for m in clean_mols(*args, **kwargs)]
    >>> show(["CC(=O)O.[Na+]", "NCc1ccccc1.Cl", "NCC(=O)O"])
    ['CC(=O)[O-]', '[NH3+]Cc1ccccc1', '[NH3+]CC(=O)[O-]']
    >>> show(["CC(=O)O.[Na+]"], protonate=False)
    ['CC(=O)O']
    >>> show(["CC(=O)O.[Na+]"], desalt=False, protonate=False)
    ['CC(=O)O.[Na+]']
    >>> show(["", None, "not-a-smiles"])
    [None, None, None]

    """
    from rdkit import RDLogger
    from rdkit.Chem import MolFromSmiles

    smiles = list(smiles)
    RDLogger.DisableLog("rdApp.*")
    try:
        if not (desalt or protonate):
            mols = [
                MolFromSmiles(s) if isinstance(s, str) and len(s) > 0 else None
                for s in smiles
            ]
        else:
            clean_mol, sanitize_smiles_to_mol = _import_schemist()
            mols = []
            for s in smiles:
                group = None
                if isinstance(s, str) and len(s) > 0:
                    try:
                        group = clean_mol(
                            sanitize_smiles_to_mol(s),
                            skip_desalt=not desalt,
                            protonate=protonate,
                            protonation_pH=protonation_pH,
                        )
                    except ImportError:
                        raise
                    except Exception:
                        group = None
                mols.append(None if group is None else group.parent)
    finally:
        RDLogger.EnableLog("rdApp.*")
    return [m if m is not None and m.GetNumAtoms() > 0 else None for m in mols]


def clean_smiles(
    smiles: Iterable[str | None],
    desalt: bool = True,
    protonate: bool = True,
    protonation_pH: float = DEFAULT_PH,
) -> list[str | None]:
    """Canonical SMILES after :func:`clean_mols`; `None` where cleaning failed.

    Examples
    ========
    >>> clean_smiles(["OC(=O)C.[Na+].[Cl-]", "C(C)O", "nonsense"])
    ['CC(=O)[O-]', 'CCO', None]

    """
    from rdkit.Chem import MolToSmiles
    return [
        None if m is None else MolToSmiles(m)
        for m in clean_mols(
            smiles,
            desalt=desalt,
            protonate=protonate,
            protonation_pH=protonation_pH,
        )
    ]


@register_function("clean-smiles")
def CleanSmiles(
    desalt: bool = True,
    protonate: bool = True,
    protonation_pH: float = DEFAULT_PH,
) -> Callable:
    """Canonical, optionally desalted and protonated SMILES, as a transform.

    Molecular featurizers use their input SMILES as given, so chain this
    before them to clean, e.g. `("smiles", ["clean-smiles", "chemeleon"])`.
    Branches that start with the same `clean-smiles` step share one cleaned
    column, so cleaning runs once per batch.

    Parameters
    ==========
    desalt : bool
        Keep only the parent fragment (schemist / RDKit `FragmentParent`).
        Default: `True`.
    protonate : bool
        Set the dominant protonation state at `protonation_pH` (schemist /
        Dimorphite-DL). Default: `True`.
    protonation_pH : float
        pH for `protonate`. Default: 7.4.

    Returns
    =======
    Callable
        Function mapping (data, input_column) → list of SMILES, with `None`
        where the input is missing or cannot be parsed.

    Examples
    ========
    >>> CleanSmiles()({"s": ["OC(=O)C.[Na+]", "C(C)O", "nonsense"]}, "s")
    ['CC(=O)[O-]', 'CCO', None]

    """
    def _clean_smiles(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> list[str | None]:
        return clean_smiles(
            data[input_column],
            desalt=desalt,
            protonate=protonate,
            protonation_pH=protonation_pH,
        )

    return _clean_smiles
