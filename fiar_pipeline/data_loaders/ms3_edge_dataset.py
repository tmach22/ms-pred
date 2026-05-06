"""
MS3EdgeDataset
==============
Maps empirical MS³ edges (SMILES + parent_mz + child_mz + NCE) from the
MSnLib ms3_edge_labels.feather into FragGNN-compatible training examples.

Design
------
For each MS³ edge (parent_mz → child_mz):
  1. Build a FragmentEngine for the molecule SMILES.
  2. Enumerate all fragments via engine.generate_fragments().
  3. Find the fragment whose base mass best matches parent_mz (within
     PARENT_TOL Da).  This fragment plays the role of the "MS2 precursor
     being fragmented."
  4. Find all sub-fragments of the parent fragment whose mass is within
     WINDOW_DA of child_mz.  For each match, assign a Gaussian weight
         w = exp(-((mass_fc - child_mz)² / (2 * SIGMA_DA²)))
     to every atom that "leaves" (present in parent but absent in child).
  5. Build the per-atom soft target vector by taking the element-wise max
     over all matching child fragments.
  6. Package the result in the dict format expected by GenDataset.collate_fn.

Items where no parent fragment is found within tolerance, or where the
fragment has no sub-fragments matching child_mz, are returned as None and
filtered by the collate function.

Compatible with GenDataset.collate_fn — the returned batch dict has the
same keys: names, root_reprs, frag_graphs, targ_atoms, frag_atoms, inds,
broken_bonds, adducts, collision_engs, precursor_mzs, root_form_vecs,
frag_form_vecs.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import dgl
from torch.utils.data import Dataset

# ── Repo path injection ────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import ms_pred.common as common
import ms_pred.magma.fragmentation as fragmentation
import ms_pred.magma.run_magma as run_magma
from ms_pred.dag_pred.dag_data import TreeProcessor, GenDataset

# ── Constants ──────────────────────────────────────────────────────────────────
FRAGMENT_ENGINE_PARAMS: dict = run_magma.FRAGMENT_ENGINE_PARAMS
SIGMA_DA:    float = 0.8   # Gaussian soft-target width (Da)
PARENT_TOL:  float = 1.5   # tolerance for parent_mz mass matching (Da)
WINDOW_DA:   float = 2.0   # search window for child fragment candidates (Da)


# ── Lazy TreeProcessor singleton ───────────────────────────────────────────────
_TREE_PROCESSOR: Optional[TreeProcessor] = None


def _get_tree_processor() -> TreeProcessor:
    global _TREE_PROCESSOR
    if _TREE_PROCESSOR is None:
        _TREE_PROCESSOR = TreeProcessor(root_encode="gnn", pe_embed_k=0)
    return _TREE_PROCESSOR


# ── Per-item construction ──────────────────────────────────────────────────────

def _gaussian_weight(mass: float, target: float, sigma: float = SIGMA_DA) -> float:
    return math.exp(-((mass - target) ** 2) / (2.0 * sigma ** 2))


def build_ms3_item(
    smiles: str,
    parent_mz: float,
    child_mz: float,
    nce: float,
    tree_processor: TreeProcessor,
) -> Optional[dict]:
    """Build one GenDataset-compatible dict from an MS3 edge.

    Returns None if:
      - The SMILES is invalid.
      - No fragment matches parent_mz within PARENT_TOL Da.
      - No sub-fragment of the parent fragment matches child_mz within WINDOW_DA Da.
    """
    # ── 1. Build and enumerate fragment engine ─────────────────────────────────
    try:
        engine = fragmentation.FragmentEngine(
            mol_str=smiles,
            mol_str_type="smiles",
            **FRAGMENT_ENGINE_PARAMS,
        )
        if engine.mol is None:
            return None
        engine.generate_fragments()
    except Exception:
        return None

    # ── 2. Find best-matching parent fragment (mass ≈ parent_mz) ──────────────
    best_fp_entry = None
    best_fp_delta = PARENT_TOL  # strictly less than this to qualify

    for fe in engine.frag_to_entry.values():
        delta = abs(fe["base_mass"] - parent_mz)
        if delta < best_fp_delta:
            best_fp_delta = delta
            best_fp_entry = fe

    if best_fp_entry is None:
        return None

    fp_int = best_fp_entry["frag"]

    # ── 3. Build Gaussian soft targets over parent-fragment atoms ──────────────
    # targ_accumulator[a] = max Gaussian weight across all matching child frags
    # where atom a leaves (present in parent but absent in child).
    targ_accumulator = np.zeros(engine.natoms, dtype=np.float32)

    for fe in engine.frag_to_entry.values():
        fc_int = fe["frag"]
        if fc_int == fp_int:
            continue
        if (fc_int & fp_int) != fc_int:  # not a subset of parent
            continue
        mass_fc = fe["base_mass"]
        if abs(mass_fc - child_mz) > WINDOW_DA:
            continue
        weight = _gaussian_weight(mass_fc, child_mz)
        if weight < 1e-4:
            continue
        # Leaving atoms = in parent (fp_int) but not in child (fc_int)
        leaving_int = fp_int & (~fc_int)
        for a in range(engine.natoms):
            if leaving_int & (1 << a):
                targ_accumulator[a] = max(targ_accumulator[a], float(weight))

    if targ_accumulator.max() == 0.0:
        return None

    # ── 4. Featurize parent fragment as a DGL graph ────────────────────────────
    try:
        frag_dict = tree_processor.featurize_frag(fp_int, engine)
    except Exception:
        return None

    graph:      dgl.DGLGraph = frag_dict["graph"]
    new_to_old: np.ndarray   = frag_dict["new_to_old"]  # graph_node → orig_atom
    frag_form:  str          = frag_dict["form"]

    # Map soft targets from original atom space into graph node space
    targ_vec = torch.from_numpy(targ_accumulator[new_to_old]).float()

    # ── 5. Root representation: full molecule graph ────────────────────────────
    root_frag_int = engine.get_root_frag()
    try:
        root_dict = tree_processor.featurize_frag(root_frag_int, engine)
    except Exception:
        return None
    root_repr: dgl.DGLGraph = root_dict["graph"]

    # ── 6. Formula encoding ────────────────────────────────────────────────────
    form_vec = common.formula_to_dense(frag_form)
    root_form_str = common.form_from_smi(smiles)
    root_form_vec = common.formula_to_dense(root_form_str)

    # ── 7. Assemble output dict (GenDataset.collate_fn compatible) ─────────────
    return {
        # GenDataset keys
        "name":           smiles,
        "root_repr":      root_repr,
        "dgl_frags":      [graph],
        "targs":          [targ_vec],
        "max_broken":     [int(best_fp_entry["max_broken"])],
        "form_vecs":      np.array([form_vec]),
        "root_form_vec":  root_form_vec,
        # Batch-scalar fields
        "collision_energy": float(nce * 50.0),   # NCE → eV for embed_collision
        "precursor":        float(parent_mz),    # MS2 precursor m/z
        "adduct":           0,                   # unknown — not in MSnLib schema
    }


# ── Dataset class ──────────────────────────────────────────────────────────────

class MS3EdgeDataset(Dataset):
    """PyTorch Dataset wrapping ms3_edge_labels.feather as FragGNN examples.

    Each row of the feather file yields at most one training item (it may be
    None if mass matching fails).  The provided collate_fn filters None items
    and calls GenDataset.collate_fn on the remainder.

    Args:
        feather_path: Path to ms3_edge_labels.feather produced by
            fiar_pipeline/scripts/preprocess_msnlib.py.
        max_mol_size: Maximum heavy-atom count.  Molecules above this limit
            are dropped to keep fragmentation tractable (O(2^natoms) worst
            case before depth/bond limits kick in).
        ion_mode: If given, filter to 'positive' or 'negative' rows only.
    """

    def __init__(
        self,
        feather_path: str,
        max_mol_size: int = 80,
        ion_mode: Optional[str] = None,
    ):
        super().__init__()
        df = pd.read_feather(feather_path)

        if ion_mode is not None:
            df = df[df["ion_mode"] == ion_mode].reset_index(drop=True)

        # Filter oversized molecules before any fragmentation work
        try:
            from rdkit import Chem

            def _heavy_atom_count(smi: str) -> int:
                mol = Chem.MolFromSmiles(smi)
                return mol.GetNumAtoms() if mol is not None else 9999

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df["_na"] = df["smiles"].map(_heavy_atom_count)

            df = df[df["_na"] <= max_mol_size].drop(columns=["_na"])
        except ImportError:
            pass  # RDKit not available; skip size filter

        self.df = df.reset_index(drop=True)
        self.tree_processor = _get_tree_processor()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Optional[dict]:
        row = self.df.iloc[idx]
        return build_ms3_item(
            smiles    = str(row["smiles"]),
            parent_mz = float(row["parent_mz"]),
            child_mz  = float(row["child_mz"]),
            nce       = float(row["nce"]),
            tree_processor = self.tree_processor,
        )

    @staticmethod
    def collate_fn(input_list: list) -> Optional[dict]:
        """Filter None items then delegate to GenDataset.collate_fn."""
        valid = [x for x in input_list if x is not None]
        if not valid:
            return None
        return GenDataset.collate_fn(valid)
