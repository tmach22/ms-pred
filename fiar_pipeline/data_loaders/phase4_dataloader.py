"""
Phase 4 Dataloader — Sinkhorn-FiLM Upgrades
======================================================
Extends SiameseFragmentDataset (Phase 3) to provide the specific physical 
priors required by the Sinkhorn Optimal Transport architecture.

REMOVED: Heavy 3D ETKDGv3 conformer generation.
ADDED: Instantaneous 2D thermodynamic extraction.

New per-sample tensors:
----------------------
frag_mass_fractions [K]    : m_i / M_precursor (Used for Sinkhorn Marginals)
thermo_state        [K, 4] : [mass_fraction, formal_charge, quat_N, NCE]
                             (Fed directly into the FiLM Modulator MLP)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from rdkit import Chem, RDLogger

# ── Import Phase 3 dataloader ─────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from fiar_pipeline.data_loaders.siamese_fragment_dataloader import (  # noqa: E402
    SiameseFragmentDataset,
    _pad_1d,
)

RDLogger.DisableLog("rdApp.*")

# ── Feature Helpers ───────────────────────────────────────────────────────────

def _mass_fractions(frag_masses: torch.Tensor, prec_mz: float) -> torch.Tensor:
    """Compute m_i / M_precursor for each fragment slot."""
    denom = max(float(prec_mz), 1.0)
    return (frag_masses / denom).clamp(0.0, 1.0)

def _extract_mol_thermo(smiles: Optional[str]) -> tuple[float, float]:
    """Rapidly extracts 2D thermodynamic surrogates (Charge, Quat-N)."""
    formal_charge = 0.0
    has_quat_n    = 0.0

    if smiles and isinstance(smiles, str):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fc = sum(a.GetFormalCharge() for a in mol.GetAtoms())
                formal_charge = float(np.clip(fc / 4.0, -1.0, 1.0))
                for atom in mol.GetAtoms():
                    if (atom.GetAtomicNum() == 7 and atom.GetFormalCharge() > 0 
                        and atom.GetTotalDegree() == 4 and atom.GetTotalNumHs() == 0):
                        has_quat_n = 1.0
                        break
        except Exception:
            pass

    return formal_charge, has_quat_n


# ── Dataset ───────────────────────────────────────────────────────────────────

class SinkhornFragmentDataset(SiameseFragmentDataset):
    def __getitem__(self, idx: int):
        # Delegate to Phase 3
        item = super().__getitem__(idx)
        if item is None:
            return None

        graph_A, graph_B, target_sim, target_label = item

        row    = self.pairs_df.iloc[idx]
        spec_A = row["name_main"]
        spec_B = row["name_sub"]
        smi_A  = self.spec_to_smiles.get(spec_A)
        smi_B  = self.spec_to_smiles.get(spec_B)
        K      = self.max_k

        # ── 1. Sinkhorn Marginals (Mass Fractions) ──
        mz_A = self.spec_to_prec_mz.get(spec_A, 500.0)
        mz_B = self.spec_to_prec_mz.get(spec_B, 500.0)
        
        mf_A = _mass_fractions(graph_A.frag_masses, mz_A)
        mf_B = _mass_fractions(graph_B.frag_masses, mz_B)
        
        graph_A.frag_mass_fractions = mf_A
        graph_B.frag_mass_fractions = mf_B

        # ── 2. Thermodynamic States (For FiLM) ──
        fc_A, qn_A = _extract_mol_thermo(smi_A)
        fc_B, qn_B = _extract_mol_thermo(smi_B)

        # Extract normalized collision energy (defaults to 0.5 if missing)
        nce_A = getattr(graph_A, "frag_nce", torch.tensor(0.5)).item() if hasattr(graph_A, "frag_nce") else 0.5
        nce_B = getattr(graph_B, "frag_nce", torch.tensor(0.5)).item() if hasattr(graph_B, "frag_nce") else 0.5

        # Build [K, 4] tensor: [mass_fraction, charge, quat_N, NCE]
        thermo_A = torch.zeros(K, 4, dtype=torch.float32)
        thermo_A[:, 0] = mf_A
        thermo_A[:, 1] = fc_A
        thermo_A[:, 2] = qn_A
        thermo_A[:, 3] = nce_A
        graph_A.thermo_state = thermo_A

        thermo_B = torch.zeros(K, 4, dtype=torch.float32)
        thermo_B[:, 0] = mf_B
        thermo_B[:, 1] = fc_B
        thermo_B[:, 2] = qn_B
        thermo_B[:, 3] = nce_B
        graph_B.thermo_state = thermo_B

        return graph_A, graph_B, target_sim, target_label


# ── Collate ───────────────────────────────────────────────────────────────────

def sinkhorn_collate_fn(batch):
    """Collate function for SinkhornFragmentDataset."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None, None

    from phase2_dataloader import phase2_collate_fn  # type: ignore  # noqa: PLC0415

    graphs_A       = [b[0] for b in batch]
    graphs_B       = [b[1] for b in batch]
    targets_sim    = torch.stack([b[2] for b in batch])
    targets_label  = torch.stack([b[3] for b in batch])

    batch_A = phase2_collate_fn(graphs_A)
    batch_B = phase2_collate_fn(graphs_B)

    if isinstance(batch_A, dict) and isinstance(batch_B, dict):
        # ── Phase 3 Padding ──
        for key in ("node_masses", "node_electronics"):
            batch_A[key] = _pad_1d(graphs_A, key)
            batch_B[key] = _pad_1d(graphs_B, key)

        if hasattr(graphs_A[0], "frag_fps"):
            for key in ("frag_fps", "frag_masses", "frag_probs"):
                batch_A[key] = torch.stack([getattr(g, key) for g in graphs_A])
                batch_B[key] = torch.stack([getattr(g, key) for g in graphs_B])

        # ── Phase 4 Sinkhorn/FiLM Tensors (No Padding Needed) ──
        for key in ("frag_mass_fractions", "thermo_state"):
            if hasattr(graphs_A[0], key):
                batch_A[key] = torch.stack([getattr(g, key) for g in graphs_A])
                batch_B[key] = torch.stack([getattr(g, key) for g in graphs_B])

    return batch_A, batch_B, targets_sim, targets_label