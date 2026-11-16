"""3D conformers as a cacheable, shareable pipeline step."""

from collections.abc import Callable, Iterable, Mapping

from carabiner import print_err

from .registry import register_function


def _import_schemist_conformers():
    try:
        from schemist.conformers import embed_conformers, validate_conformer_args
    except ImportError as e:
        raise ImportError(
            f"The `conformers` transform needs schemist>=0.1.4 ({e}). "
            "Try `pip install aspect-data[chem]`."
        ) from e
    return embed_conformers, validate_conformer_args


@register_function("conformers")
def Conformers(
    n_conformers: int = 1,
    seed: int = 42,
    etkdg_version: int = 3,
    prune_rms_thresh: float = -1.,
    minimize: bool = True,
    max_iters: int = 500,
    n_threads: int = 1,
) -> Callable:
    """3D conformers with hydrogens, from SMILES, via schemist.

    SMILES are used as given (chain `clean-smiles` first to clean them);
    conformers depend on atom order. Conformers are embedded with ETKDG and
    relaxed with MMFF94 (see `schemist.conformers.embed_conformers`).
    Without pruning, every molecule has exactly `n_conformers`, or none. The
    defaults reproduce the Atom-JEPA protocol, including that rule, so with
    the same RDKit version `("smiles", ["conformers", "atom-jepa"])` matches
    upstream Atom-JEPA (tested bit-for-bit with RDKit 2026.09 and atom-jepa
    0.4.0). Conformers can differ across RDKit versions; the RDKit version is
    part of aspect's cache key.

    Branches starting with the same `conformers` step share it, so, for
    example, Atom-JEPA and Atom-JEPA-Bio embeddings use one set of conformers.
    The output is a column of structs, which Arrow and Parquet store exactly.

    Parameters
    ==========
    n_conformers : int
        Conformers per molecule. Default: 1.
    seed : int
        ETKDG random seed. Default: 42.
    etkdg_version : int
        ETKDG version, 2, 3 or 4. Default: 3.
    prune_rms_thresh : float
        RMSD (Å) for pruning duplicate conformers; negative disables pruning
        (then every molecule has exactly `n_conformers`). Default: -1.
    minimize : bool
        Relax with MMFF94. Default: `True`.
    max_iters : int
        Maximum MMFF94 steps. Default: 500.
    n_threads : int
        Threads per molecule for embedding and minimization. Default: 1.

    Returns
    =======
    Callable
        Function mapping (data, input_column) → list, per molecule, of
        {"atomic_numbers": int8 [N], "coordinates": float32 [C, N, 3] (Å)},
        or `None` where the SMILES is missing or invalid, or conformers could
        not be generated. To flag these, chain `missing` after this step,
        e.g. `("smiles", ["conformers", "missing"])`.

    """
    embed_conformers, validate_conformer_args = _import_schemist_conformers()
    # Configuration errors are raised here, so per-row errors below are
    # always about molecules.
    validate_conformer_args(
        n_conformers, seed, etkdg_version, prune_rms_thresh, minimize, max_iters, n_threads
    )
    kwargs = dict(
        n_conformers=n_conformers,
        seed=seed,
        etkdg_version=etkdg_version,
        prune_rms_thresh=prune_rms_thresh,
        minimize=minimize,
        max_iters=max_iters,
        n_threads=n_threads,
    )

    def _conformers(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> list[dict | None]:
        out, n_invalid, n_failed = [], 0, 0
        for smiles in data[input_column]:
            try:
                atomic_numbers, coordinates = embed_conformers(smiles, **kwargs)
            except ValueError:  # missing or invalid SMILES
                n_invalid += 1
                out.append(None)
            except RuntimeError:  # valid SMILES, conformers failed
                n_failed += 1
                out.append(None)
            else:
                if prune_rms_thresh < 0 and len(coordinates) != n_conformers:
                    n_failed += 1  # partial embedding
                    out.append(None)
                    continue
                out.append({
                    "atomic_numbers": atomic_numbers,
                    "coordinates": coordinates,
                })
        if n_invalid > 0 or n_failed > 0:
            print_err(
                f"[WARN] conformers: {n_invalid} missing or invalid SMILES, and "
                f"{n_failed} valid SMILES whose conformers could not be "
                "generated, given null conformers."
            )
        return out

    return _conformers
