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

Performance caching (per worker process)
-----------------------------------------
generate_fragments() is O(2^natoms) bounded by max_tree_depth and can take
100–500 ms per molecule.  Since multiple rows share the same SMILES, four
module-level caches eliminate redundant work:

  _ENGINE_CACHE[smiles]           FragmentEngine with frag_to_entry populated
  _MASS_CACHE[smiles]             (masses_arr, frag_ints_arr, max_broken_arr)
                                   pre-extracted numpy arrays for fast mass
                                   matching — avoids dict iteration every call
  _ROOT_GRAPH_CACHE[smiles]       Root-molecule DGL graph (PE-embedded)
  _FRAG_GRAPH_CACHE[(smi,fp_int)] (graph, new_to_old, form) parent-fragment
                                   DGL graph (PE-embedded)

Each DataLoader worker process has its own copy of these caches.  With 16
workers the typical 38-edges-per-SMILES ratio means ~97% cache hit rate
(one cold miss, 37 warm hits) — reducing effective cost to nearly zero for
repeated SMILES within a worker's lifetime.

Compatible with GenDataset.collate_fn — the returned batch dict has the
same keys: names, root_reprs, frag_graphs, targ_atoms, frag_atoms, inds,
broken_bonds, adducts, collision_engs, precursor_mzs, root_form_vecs,
frag_form_vecs.
"""

from __future__ import annotations

import math
import sys
import threading
import warnings
from pathlib import Path
from typing import Optional, Tuple

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
ENGINE_TIMEOUT_S: float = 5.0  # per-molecule generate_fragments() timeout (seconds)


# ── Per-worker process caches (not shared across processes) ───────────────────
_ENGINE_CACHE:     dict[str, fragmentation.FragmentEngine] = {}
_MASS_CACHE:       dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_ROOT_GRAPH_CACHE: dict[str, dgl.DGLGraph] = {}
_FRAG_GRAPH_CACHE: dict[Tuple[str, int], Tuple[dgl.DGLGraph, np.ndarray, str]] = {}


def _clear_caches() -> None:
    """Empty all per-worker caches (useful in tests)."""
    _ENGINE_CACHE.clear()
    _MASS_CACHE.clear()
    _ROOT_GRAPH_CACHE.clear()
    _FRAG_GRAPH_CACHE.clear()


# ── TreeProcessor factory ──────────────────────────────────────────────────────
_TP_CACHE: dict = {}


def get_tree_processor(
    pe_embed_k: int = 0,
    add_hs: bool = False,
    embed_elem_group: bool = False,
) -> TreeProcessor:
    """Return a cached TreeProcessor for the given featurisation config.

    These kwargs must match the hparams of the FragGNN checkpoint that will
    consume the graphs produced by this processor.
    """
    key = (pe_embed_k, add_hs, embed_elem_group)
    if key not in _TP_CACHE:
        _TP_CACHE[key] = TreeProcessor(
            root_encode="gnn",
            pe_embed_k=pe_embed_k,
            add_hs=add_hs,
            embed_elem_group=embed_elem_group,
        )
    return _TP_CACHE[key]


def _get_tree_processor() -> TreeProcessor:
    return get_tree_processor()


# ── Engine + graph cache helpers ───────────────────────────────────────────────

def _get_engine(smiles: str) -> Optional[fragmentation.FragmentEngine]:
    """Return cached FragmentEngine (with fragments enumerated), or None.

    generate_fragments() is O(2^natoms) in the worst case and can hang for
    minutes on molecules with many tautomers.  A daemon thread enforces a
    30-second hard cap; molecules that exceed it are cached as None so
    subsequent calls for the same SMILES skip immediately.
    """
    if smiles in _ENGINE_CACHE:
        return _ENGINE_CACHE[smiles]

    result_box: list = []
    exc_box:    list = []

    def _run() -> None:
        try:
            engine = fragmentation.FragmentEngine(
                mol_str=smiles,
                mol_str_type="smiles",
                **FRAGMENT_ENGINE_PARAMS,
            )
            if engine.mol is None:
                result_box.append(None)
                return
            engine.generate_fragments()
            result_box.append(engine)
        except Exception as exc:
            exc_box.append(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=ENGINE_TIMEOUT_S)

    if t.is_alive():
        # Thread is still running — molecule is too slow; cache miss → skip
        _ENGINE_CACHE[smiles] = None
        return None

    if exc_box or not result_box:
        _ENGINE_CACHE[smiles] = None
        return None

    _ENGINE_CACHE[smiles] = result_box[0]
    return _ENGINE_CACHE[smiles]


def _get_mass_arrays(
    smiles: str,
    engine: fragmentation.FragmentEngine,
) -> Tuple[np.ndarray, list, np.ndarray]:
    """Return (masses_f32, frag_ints_list, max_broken_i32) for mass matching.

    frag_ints is a plain Python list (not a numpy array) because fragment
    bitmasks can exceed 63 bits for molecules with >63 atoms, overflowing
    np.int64.  All bitwise operations on these values must remain in Python.
    """
    if smiles not in _MASS_CACHE:
        entries     = list(engine.frag_to_entry.values())
        masses      = np.array([e["base_mass"]  for e in entries], dtype=np.float32)
        frag_ints   = [e["frag"]                for e in entries]   # Python big-ints
        max_brokens = np.array([e["max_broken"] for e in entries], dtype=np.int32)
        _MASS_CACHE[smiles] = (masses, frag_ints, max_brokens)
    return _MASS_CACHE[smiles]


def _get_root_graph(
    smiles: str,
    engine: fragmentation.FragmentEngine,
    tree_processor: TreeProcessor,
) -> Optional[dgl.DGLGraph]:
    """Return cached PE-embedded root-molecule DGL graph."""
    if smiles not in _ROOT_GRAPH_CACHE:
        root_frag_int = engine.get_root_frag()
        try:
            root_dict = tree_processor.featurize_frag(root_frag_int, engine)
            g = root_dict["graph"]
            if tree_processor.pe_embed_k > 0:
                tree_processor.add_pe_embed(g)
            _ROOT_GRAPH_CACHE[smiles] = g
        except Exception:
            return None
    return _ROOT_GRAPH_CACHE[smiles]


def _get_frag_graph(
    smiles: str,
    fp_int: int,
    engine: fragmentation.FragmentEngine,
    tree_processor: TreeProcessor,
) -> Optional[Tuple[dgl.DGLGraph, np.ndarray, str]]:
    """Return cached (graph, new_to_old, form) for a parent fragment."""
    key = (smiles, fp_int)
    if key not in _FRAG_GRAPH_CACHE:
        try:
            frag_dict = tree_processor.featurize_frag(fp_int, engine)
            g         = frag_dict["graph"]
            new_to_old = frag_dict["new_to_old"]
            form       = frag_dict["form"]
            if tree_processor.pe_embed_k > 0:
                tree_processor.add_pe_embed(g)
            _FRAG_GRAPH_CACHE[key] = (g, new_to_old, form)
        except Exception:
            return None
    return _FRAG_GRAPH_CACHE[key]


# ── Gaussian weight ────────────────────────────────────────────────────────────

def _gaussian_weight(mass: float, target: float, sigma: float = SIGMA_DA) -> float:
    return math.exp(-((mass - target) ** 2) / (2.0 * sigma ** 2))


# ── Per-item construction ──────────────────────────────────────────────────────

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
      - No sub-fragment of the parent fragment matches child_mz within WINDOW_DA.
    """
    # ── 1. Get engine (cached after first call per SMILES per worker) ──────────
    engine = _get_engine(smiles)
    if engine is None:
        return None

    # ── 2. Fast vectorised parent-fragment mass matching ──────────────────────
    masses, frag_ints, max_brokens = _get_mass_arrays(smiles, engine)
    # masses / max_brokens are numpy arrays; frag_ints is a plain Python list

    deltas = np.abs(masses - parent_mz)
    best_idx = int(np.argmin(deltas))
    if deltas[best_idx] >= PARENT_TOL:
        return None

    fp_int    = frag_ints[best_idx]   # Python int (may be >63 bits)
    fp_broken = int(max_brokens[best_idx])

    # ── 3. Build Gaussian soft targets over parent-fragment atoms ──────────────
    child_diffs = np.abs(masses - child_mz)
    # Candidate child indices (mass filter only; subset check done in Python below)
    child_candidate_idxs = np.where(child_diffs < WINDOW_DA)[0]

    targ_accumulator = np.zeros(engine.natoms, dtype=np.float32)

    for ci in child_candidate_idxs:
        fc_int   = frag_ints[ci]   # Python int
        if fc_int == fp_int:
            continue
        mass_fc  = float(masses[ci])
        # Must be a proper subset of parent
        if (fc_int & fp_int) != fc_int:
            continue
        weight = _gaussian_weight(mass_fc, child_mz)
        if weight < 1e-4:
            continue
        leaving_int = fp_int & (~fc_int)
        for a in range(engine.natoms):
            if leaving_int & (1 << a):
                if weight > targ_accumulator[a]:
                    targ_accumulator[a] = weight

    if targ_accumulator.max() == 0.0:
        return None

    # ── 4. Get parent fragment DGL graph (cached after first call) ─────────────
    frag_cached = _get_frag_graph(smiles, fp_int, engine, tree_processor)
    if frag_cached is None:
        return None
    graph, new_to_old, frag_form = frag_cached

    # Map soft targets from original atom space into graph node space
    targ_vec = torch.from_numpy(targ_accumulator[new_to_old]).float()

    # ── 5. Get root DGL graph (cached after first call) ───────────────────────
    root_repr = _get_root_graph(smiles, engine, tree_processor)
    if root_repr is None:
        return None

    # ── 6. Formula encoding ────────────────────────────────────────────────────
    form_vec      = common.formula_to_dense(frag_form)
    root_form_str = common.form_from_smi(smiles)
    root_form_vec = common.formula_to_dense(root_form_str)

    # ── 7. Assemble output dict (GenDataset.collate_fn compatible) ─────────────
    return {
        "name":           smiles,
        "root_repr":      root_repr,
        "dgl_frags":      [graph],
        "targs":          [targ_vec],
        "max_broken":     [fp_broken],
        "form_vecs":      np.array([form_vec]),
        "root_form_vec":  root_form_vec,
        "collision_energy": float(nce * 50.0),
        "precursor":        float(parent_mz),
        "adduct":           0,
    }


# ── Dataset class ──────────────────────────────────────────────────────────────

class MS3EdgeDataset(Dataset):
    """PyTorch Dataset wrapping ms3_edge_labels.feather as FragGNN examples.

    Each row of the feather file yields at most one training item (may be
    None if mass matching fails).  The provided collate_fn filters None items
    and calls GenDataset.collate_fn on the remainder.

    Performance note: rows are sorted by SMILES before storage so that
    consecutive indices within each DataLoader worker share the same molecule,
    maximising per-worker cache hit rates.

    Args:
        feather_path: Path to ms3_edge_labels.feather.
        max_mol_size: Maximum heavy-atom count.  Molecules above this limit
            are dropped to keep fragmentation tractable.
        ion_mode:     Filter to 'positive' or 'negative' (or None for both).
        tree_processor_kwargs: Kwargs forwarded to ``get_tree_processor()``.
            Must match the FragGNN checkpoint hparams.  Critical keys:
              ``pe_embed_k``, ``add_hs``, ``embed_elem_group``.
    """

    def __init__(
        self,
        feather_path: str,
        max_mol_size: int = 80,
        ion_mode: Optional[str] = None,
        tree_processor_kwargs: Optional[dict] = None,
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
            pass

        # Sort by SMILES: consecutive rows with the same molecule share a
        # worker-local cache slot, reducing generate_fragments() calls from
        # O(n_rows) to O(n_unique_smiles) per worker.
        df = df.sort_values("smiles").reset_index(drop=True)

        self.df = df
        self.tree_processor = get_tree_processor(**(tree_processor_kwargs or {}))

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
