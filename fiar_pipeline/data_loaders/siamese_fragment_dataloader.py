"""
SiameseFragmentDataset
======================
Adapted from SpectralSimilarityPredictor/data_loaders/fiar/phase3_multitask_dataloader.py
for use inside the ms-pred repo.

Changes vs the external version
--------------------------------
- All hardcoded absolute paths replaced with constructor arguments.
- Fragment cache loading added as a first-class feature (not patched in).
- phase2_dataloader import is handled via sys.path injection at __init__ time
  because Phase2EdgeDataset lives in SpectralSimilarityPredictor, not ms-pred.
  Pass `phase2_loader_dir` to point at data_loaders/fiar/ in the old repo.
- No wandb dependency here — kept in the training script.

Usage
-----
from fiar_pipeline.data_loaders.siamese_fragment_dataloader import (
    SiameseFragmentDataset, siamese_frag_collate_fn
)

dataset = SiameseFragmentDataset(
    feather_path   = "path/to/pairs.feather",
    graphs_path    = "path/to/phase3_graphs_lpe.pt",
    spec_df_path   = "path/to/spec_df.pkl",
    mol_df_path    = "path/to/mol_df.pkl",
    phase2_loader_dir = "path/to/SpectralSimilarityPredictor/data_loaders/fiar",
    fragment_cache_path = "fiar_pipeline/data/fragment_cache.pt",
    max_k          = 100,
    morgan_nbits   = 2048,
)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdPartialCharges
from torch.utils.data import Dataset

RDLogger.DisableLog("rdApp.*")


def _inject_phase2_path(phase2_loader_dir: str) -> None:
    """Add SpectralSimilarityPredictor's data_loaders/fiar to sys.path once."""
    abs_dir = str(Path(phase2_loader_dir).resolve())
    if abs_dir not in sys.path:
        sys.path.insert(0, abs_dir)


class SiameseFragmentDataset(Dataset):
    """
    Paired Siamese dataset that serves three feature modalities per sample:
      1. Phase2EdgeDataset graph objects    (existing backbone)
      2. Per-node Gasteiger charge + mass   (thermodynamic adapter)
      3. Top-K ICEBERG fragment FPs/masses  (new, from fragment cache)

    Parameters
    ----------
    feather_path        : .feather with name_main, name_sub, label, entropy_similarity
    graphs_path         : .pt file produced by Phase2EdgeDataset preprocessing
    spec_df_path        : spec_df.pkl  (spec_id → mol_id, prec_type, ...)
    mol_df_path         : mol_df.pkl   (mol_id → smiles, ...)
    phase2_loader_dir   : path to SpectralSimilarityPredictor/data_loaders/fiar/
    fragment_cache_path : .pt from extract_fragment_cache.py;  None = disabled
    max_k               : max fragment slots (must equal top_k used in ETL)
    morgan_nbits        : fingerprint length (must match ETL setting)
    """

    def __init__(
        self,
        feather_path: str,
        graphs_path: str,
        spec_df_path: str,
        mol_df_path: str,
        phase2_loader_dir: str,
        fragment_cache_path: Optional[str] = None,
        max_k: int = 100,
        morgan_nbits: int = 2048,
    ):
        # ── Load pairs ────────────────────────────────────────────────────────
        print(f"[DataLoader] Pairs: {feather_path}")
        self.pairs_df = pd.read_feather(feather_path)

        # ── Phase2EdgeDataset ─────────────────────────────────────────────────
        _inject_phase2_path(phase2_loader_dir)
        try:
            from phase2_dataloader import Phase2EdgeDataset  # type: ignore
        except ImportError as e:
            raise ImportError(
                f"Cannot import Phase2EdgeDataset from {phase2_loader_dir}: {e}"
            )

        print(f"[DataLoader] Graphs: {graphs_path}")
        self.base_dataset = Phase2EdgeDataset(processed_graphs_path=graphs_path)

        self.spec_to_idx: dict = {}
        for i, g in enumerate(self.base_dataset.graphs):
            key = getattr(g, "spec_id", None)
            if key is not None:
                self.spec_to_idx[key] = i
        print(f"[DataLoader] Mapped {len(self.spec_to_idx)} spec_id → graph idx.")

        # ── Chemical metadata ─────────────────────────────────────────────────
        self.spec_to_smiles: dict = {}
        self.spec_to_mol_id: dict = {}
        self.spec_to_prec_mz: dict = {}
        self.spec_to_nce: dict = {}
        try:
            spec_df = pd.read_pickle(spec_df_path)
            mol_df  = pd.read_pickle(mol_df_path)
            mol_to_smi = dict(zip(mol_df["mol_id"].astype(int),
                                  mol_df["smiles"]))
            for _, row in spec_df.iterrows():
                sid = row["spec_id"]
                mid = int(row["mol_id"])
                self.spec_to_smiles[sid] = mol_to_smi.get(mid)
                self.spec_to_mol_id[sid] = mid
                # prec_mz and NCE for the Phase 3 metadata tensor
                # Column names follow the ETL convention (nce_updated, prec_mz)
                raw_mz = row.get("prec_mz", np.nan)
                self.spec_to_prec_mz[sid] = (
                    float(raw_mz) if not pd.isna(raw_mz) else 500.0
                )
                raw_ce = row.get("nce_updated", np.nan)
                ce_val = float(raw_ce) if not pd.isna(raw_ce) else 40.0
                self.spec_to_nce[sid] = ce_val / 50.0  # NCE = CE / 50
            print(f"[DataLoader] SMILES loaded for {len(self.spec_to_smiles)} spectra.")
        except Exception as exc:
            print(f"[DataLoader] WARNING: chemical metadata load failed: {exc}")

        # ── Fragment cache ────────────────────────────────────────────────────
        self.max_k = max_k
        self.morgan_nbits = morgan_nbits
        self.fragment_cache: Optional[dict] = None

        if fragment_cache_path and Path(fragment_cache_path).exists():
            print(f"[DataLoader] Fragment cache: {fragment_cache_path}")
            self.fragment_cache = torch.load(
                fragment_cache_path, weights_only=False
            )
            hits = sum(1 for v in self.fragment_cache.values() if v)
            print(f"[DataLoader] {len(self.fragment_cache)} entries, "
                  f"{hits} non-empty.")
        elif fragment_cache_path:
            print(f"[DataLoader] WARNING: cache not found at {fragment_cache_path}. "
                  "Fragment features disabled.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _thermodynamics(self, smiles: Optional[str], n_nodes: int):
        masses  = torch.ones(n_nodes,  dtype=torch.float32) * 12.0
        charges = torch.zeros(n_nodes, dtype=torch.float32)
        if not smiles or not isinstance(smiles, str):
            return masses, charges
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                rdPartialCharges.ComputeGasteigerCharges(mol)
                for i, atom in enumerate(mol.GetAtoms()):
                    if i >= n_nodes:
                        break
                    masses[i] = atom.GetMass()
                    c = atom.GetDoubleProp("_GasteigerCharge") \
                        if atom.HasProp("_GasteigerCharge") else 0.0
                    if np.isnan(c) or np.isinf(c):
                        c = 0.0
                    charges[i] = float(c)
        except Exception:
            pass
        return masses, charges

    def _frag_tensors(self, spec_id: str):
        """
        Returns (frag_fps, frag_masses, frag_probs) zero-padded to (max_k, *).
        Zero tensors when cache is absent or mol has no fragments.
        """
        K, D = self.max_k, self.morgan_nbits
        fps    = torch.zeros(K, D,  dtype=torch.float32)
        masses = torch.zeros(K,     dtype=torch.float32)
        probs  = torch.zeros(K,     dtype=torch.float32)

        if self.fragment_cache is None:
            return fps, masses, probs

        mol_id = self.spec_to_mol_id.get(spec_id)
        if mol_id is None:
            return fps, masses, probs

        for i, frag in enumerate(self.fragment_cache.get(mol_id, [])[:K]):
            fp = frag.get("morgan_fp")
            if fp is not None:
                fps[i] = fp if isinstance(fp, torch.Tensor) else torch.tensor(fp)
            masses[i] = float(frag.get("exact_mass", 0.0))
            probs[i]  = float(frag.get("prob_gen",   0.0))

        return fps, masses, probs

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs_df)

    def __getitem__(self, idx: int):
        row = self.pairs_df.iloc[idx]
        spec_A, spec_B = row["name_main"], row["name_sub"]

        idx_A = self.spec_to_idx.get(spec_A)
        idx_B = self.spec_to_idx.get(spec_B)
        if idx_A is None or idx_B is None:
            return None

        target_sim   = torch.tensor(row["entropy_similarity"], dtype=torch.float32)
        target_label = torch.tensor(row["label"],              dtype=torch.float32)

        graph_A = self.base_dataset[idx_A].clone()
        graph_B = self.base_dataset[idx_B].clone()
        graph_A.idx = graph_B.idx = idx

        # Thermodynamic features
        n_A = graph_A.x.size(0) if hasattr(graph_A, "x") else graph_A.num_nodes
        n_B = graph_B.x.size(0) if hasattr(graph_B, "x") else graph_B.num_nodes
        m_A, c_A = self._thermodynamics(self.spec_to_smiles.get(spec_A), n_A)
        m_B, c_B = self._thermodynamics(self.spec_to_smiles.get(spec_B), n_B)
        graph_A.node_masses     = m_A
        graph_A.node_electronics = c_A
        graph_B.node_masses     = m_B
        graph_B.node_electronics = c_B

        # Fragment cache features
        fps_A, fmass_A, fprob_A = self._frag_tensors(spec_A)
        fps_B, fmass_B, fprob_B = self._frag_tensors(spec_B)
        graph_A.frag_fps    = fps_A
        graph_A.frag_masses = fmass_A
        graph_A.frag_probs  = fprob_A
        graph_B.frag_fps    = fps_B
        graph_B.frag_masses = fmass_B
        graph_B.frag_probs  = fprob_B

        # Per-spectrum precursor m/z and NCE for Phase 3 metadata tensor
        graph_A.frag_prec_mz = torch.tensor(
            self.spec_to_prec_mz.get(spec_A, 500.0), dtype=torch.float32
        )
        graph_A.frag_nce = torch.tensor(
            self.spec_to_nce.get(spec_A, 0.8), dtype=torch.float32
        )
        graph_B.frag_prec_mz = torch.tensor(
            self.spec_to_prec_mz.get(spec_B, 500.0), dtype=torch.float32
        )
        graph_B.frag_nce = torch.tensor(
            self.spec_to_nce.get(spec_B, 0.8), dtype=torch.float32
        )

        return graph_A, graph_B, target_sim, target_label


# ── Collate ───────────────────────────────────────────────────────────────────

def _pad_1d(graphs, attr: str) -> torch.Tensor:
    tensors = [getattr(g, attr) for g in graphs]
    max_len = max(t.size(0) for t in tensors)
    out = torch.zeros(len(tensors), max_len, dtype=torch.float32)
    for i, t in enumerate(tensors):
        out[i, :t.size(0)] = t
    return out


def siamese_frag_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None, None

    # Lazy import so this module can be imported without phase2_dataloader
    from phase2_dataloader import phase2_collate_fn  # type: ignore

    graphs_A = [b[0] for b in batch]
    graphs_B = [b[1] for b in batch]
    targets_sim   = torch.stack([b[2] for b in batch])
    targets_label = torch.stack([b[3] for b in batch])

    batch_A = phase2_collate_fn(graphs_A)
    batch_B = phase2_collate_fn(graphs_B)

    if isinstance(batch_A, dict) and isinstance(batch_B, dict):
        # Thermodynamic (variable-length → pad)
        batch_A["node_masses"]      = _pad_1d(graphs_A, "node_masses")
        batch_A["node_electronics"] = _pad_1d(graphs_A, "node_electronics")
        batch_B["node_masses"]      = _pad_1d(graphs_B, "node_masses")
        batch_B["node_electronics"] = _pad_1d(graphs_B, "node_electronics")

        # Fragment cache (fixed K × D — direct stack, no padding needed)
        if hasattr(graphs_A[0], "frag_fps"):
            batch_A["frag_fps"]    = torch.stack([g.frag_fps    for g in graphs_A])
            batch_A["frag_masses"] = torch.stack([g.frag_masses for g in graphs_A])
            batch_A["frag_probs"]  = torch.stack([g.frag_probs  for g in graphs_A])
            batch_B["frag_fps"]    = torch.stack([g.frag_fps    for g in graphs_B])
            batch_B["frag_masses"] = torch.stack([g.frag_masses for g in graphs_B])
            batch_B["frag_probs"]  = torch.stack([g.frag_probs  for g in graphs_B])

        # Per-spectrum scalars for Phase 3 metadata tensor
        if hasattr(graphs_A[0], "frag_prec_mz"):
            batch_A["prec_mz"] = torch.stack([g.frag_prec_mz for g in graphs_A])
            batch_A["nce"]     = torch.stack([g.frag_nce     for g in graphs_A])
            batch_B["prec_mz"] = torch.stack([g.frag_prec_mz for g in graphs_B])
            batch_B["nce"]     = torch.stack([g.frag_nce     for g in graphs_B])

    return batch_A, batch_B, targets_sim, targets_label
