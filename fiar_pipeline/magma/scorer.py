"""
MAGMa-based fragment scorer for MS3 distillation training.

Called inside multiprocessing pool workers — all heavy imports are deferred
to function scope to avoid fork-safety issues with the parent process.
"""

import numpy as np


def magma_score_fragments(
    candidate_smiles_list: list,
    ms3_mz_arr: np.ndarray,
    ms3_int_arr: np.ndarray,
    precursor_mz: float = None,
    mass_tol_da: float = 0.5,
) -> float:
    """
    Score ICEBERG-generated fragment SMILES against observed MS3 peaks.

    Pipeline per molecule
    ---------------------
    1. Strip padding zeros from the observed MS3 m/z array.
    2. Pre-filter candidates heavier than the MS2 precursor (physically
       impossible sub-fragments — avoids wasted FragmentEngine calls).
    3. For each surviving candidate, run MAGMa's FragmentEngine to produce
       all possible sub-fragment masses (with H-shift variants).
    4. Count observed MS3 peaks explained by any sub-fragment within
       ``mass_tol_da`` Daltons.

    Parameters
    ----------
    candidate_smiles_list : list[str]
        Fragment SMILES produced by ICEBERGScalpel.extract_batch() for one
        parent molecule.
    ms3_mz_arr : np.ndarray
        Observed MS3 m/z array (may contain trailing zeros from pad_sequence).
    ms3_int_arr : np.ndarray
        Observed MS3 intensity array (parallel; kept for API symmetry, unused).
    precursor_mz : float, optional
        MS2 precursor m/z — used to pre-filter impossible fragments.
        Pass None to skip the pre-filter.
    mass_tol_da : float
        Mass matching tolerance in Daltons (default 0.5 Da).

    Returns
    -------
    float
        Fraction of observed MS3 peaks matched, in [0, 1].
    """
    # Deferred imports — keeps these out of parent-process memory before fork.
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from ms_pred.magma.fragmentation import FragmentEngine

    # Strip padding zeros from observed peaks
    observed_mz = ms3_mz_arr[ms3_mz_arr > 0]
    if len(observed_mz) == 0 or not candidate_smiles_list:
        return 0.0

    all_masses = []

    for smi in candidate_smiles_list:
        if not smi:
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # Pre-filter: a sub-fragment cannot be heavier than its MS2 precursor.
        # Allow +10 Da slack to cover adduct/isotope edge cases.
        if precursor_mz is not None and precursor_mz > 0:
            frag_mass = Descriptors.ExactMolWt(mol)
            if frag_mass > precursor_mz + 10.0:
                continue

        try:
            engine = FragmentEngine(mol_str=smi)
            engine.generate_fragments()
            _, _, _, masses, _ = engine.get_frag_masses()
            if len(masses) > 0:
                all_masses.append(masses)
        except Exception:
            continue

    if not all_masses:
        return 0.0

    all_masses = np.concatenate(all_masses)

    matched = sum(
        1 for mz in observed_mz
        if np.any(np.abs(all_masses - mz) < mass_tol_da)
    )
    return float(matched) / len(observed_mz)
