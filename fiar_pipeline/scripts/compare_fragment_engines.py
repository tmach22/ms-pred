"""
compare_fragment_engines.py
============================
Side-by-side evaluation of two fragment engines on a random subset of
validation molecules:

  · ICEBERG (FragGNN)     — pre-computed fragments from fragment_cache.pt
  · Phase 2.5 (DeterministicPhase2) — online inference using the trained
                                       deterministic student checkpoint

Metrics per engine (averaged across sampled molecules):
  1. Chemical Validity (%)   – fragment SMILES pass RDKit SanitizeMol
  2. Peak Coverage (5 ppm)   – ≥1 fragment mass matches an experimental peak
  3. Avg Fragment Atom Count – fragment size distribution
  4. Unique SMILES per mol   – structural diversity
  5. Avg Bond-break Count    – ICEBERG `max_broken`; Phase 2.5 soft-cut edges

Environment requirement — must run with MF-GPU Python 3.8 (has dgllife):
    /data/nas-gpu/wang/tmach007/massformer/y/envs/MF-GPU/bin/python3.8 \\
        fiar_pipeline/scripts/compare_fragment_engines.py

Run from repo root:
    cd /data/nas-gpu/wang/tmach007/ms-pred
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SSP  = Path("/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor")
_MF   = Path("/data/nas-gpu/wang/tmach007/massformer/src/massformer")

# ms-pred (for fiar_pipeline) + src/ (for ms_pred)
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

# SpectralSimilarityPredictor model directories
sys.path.insert(0, str(_SSP / "model" / "fiar"))
sys.path.insert(0, str(_SSP / "model" / "transport_model" / "bifurcate"))

# MassFormer package (algos1.cpython-38 .so requires Python 3.8)
sys.path.insert(0, str(_MF))

from rdkit import Chem
from rdkit.Chem import Descriptors

try:
    from gf_data_utils import gf_preprocess, collator as mf_collator
    print("[Import] MassFormer gf_data_utils  ✓")
except ImportError as e:
    print(f"[Import] FATAL — MassFormer unavailable: {e}")
    print("  Use: /data/nas-gpu/wang/tmach007/massformer/y/envs/MF-GPU/bin/python3.8")
    sys.exit(1)

try:
    from deterministic_phase2 import DeterministicPhase2
    print("[Import] DeterministicPhase2        ✓")
except ImportError as e:
    print(f"[Import] FATAL — DeterministicPhase2 unavailable: {e}")
    sys.exit(1)


# ── Graphormer-base model config (same architecture as training) ───────────────
# gf_pretrain_name='none': skip downloading pcqm4mv2 weights; we load our own.
_MODEL_CONFIG: dict = {
    "embed_types":           ["gf_v2"],
    "gf_model_name":         "graphormer_base",
    "gf_pretrain_name":      "none",
    "fix_num_pt_layers":     0,
    "reinit_num_pt_layers":  0,
    "reinit_layernorm":      False,
    "embed_dim":             -1,        # inferred as 768 from GFv2Embedder
    "embed_linear":          False,
    "ff_layer_type":         "neims",
    "ff_h_dim":              1000,
    "ff_num_layers":         4,
    "ff_skip":               True,
    "output_normalization":  "l1",
    "bidirectional_prediction": True,
    "spectrum_attention":    False,
    "gate_prediction":       False,
    "model_seed":            0,
    "dropout":               0.15,
}

_PATHS = {
    "graphs":     str(_SSP / "mass_spec_gym_data" / "phase3_graphs_lpe.pt"),
    "spec_df":    str(_SSP / "mass_spec_gym_data" / "spec_df.pkl"),
    "mol_df":     str(_SSP / "mass_spec_gym_data" / "mol_df.pkl"),
    "ckpt_p25":   str(_SSP / "trained_model" / "fiar" / "desaf_phase2_5_entropy_best.pt"),
    "ib_cache":   str(_REPO_ROOT / "fiar_pipeline" / "data" / "fragment_cache.pt"),
    
}

_PPM = 5e-6  # 5 ppm Orbitrap tolerance


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sparse_edges(mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (edge_index [2, 2*E], edge_attr_physics [2*E, 2]) for the molecule."""
    edge_idx, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j   = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt     = bond.GetBondTypeAsDouble()
        conj   = 1.0 if bond.GetIsConjugated() else 0.0
        edge_idx  += [[i, j], [j, i]]
        edge_attr += [[bt, conj], [bt, conj]]
    if edge_idx:
        return (
            torch.tensor(edge_idx,  dtype=torch.long ).t().contiguous(),
            torch.tensor(edge_attr, dtype=torch.float),
        )
    return (
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((0, 2), dtype=torch.float),
    )


def _build_p25_batch(
    smiles: str,
    device: torch.device,
    pe: Optional[torch.Tensor],
) -> Optional[dict]:
    """
    Build a DeterministicPhase2-compatible batch dict from a SMILES string.

    Layout:
      · MassFormer dense fields (x, attn_bias, spatial_pos, …) from gf_preprocess/collator
      · Sparse fields injected afterwards: edge_index, edge_attr_physics, pe
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mf_item = gf_preprocess(mol, idx=0, algos_v="algos1")
        batch   = mf_collator([mf_item])
    except Exception as exc:
        print(f"  [Phase2.5 batch] {exc}")
        return None

    ei, ea = _sparse_edges(mol)
    batch["edge_index"]        = ei.to(device)
    batch["edge_attr_physics"] = ea.to(device)

    # LPE from the precomputed graph (shape [num_nodes, 8])
    # DeterministicPhase2.forward() falls back to zeros if 'pe' is absent.
    if pe is not None:
        batch["pe"] = pe.to(device)

    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _p25_fragments(
    smiles:     str,
    S_dense:    torch.Tensor,   # [1, N, 10]
    p_keep:     torch.Tensor,   # [total_edges]
    num_nodes:  int,
) -> Tuple[List[dict], int]:
    """
    Hard-assign atoms to fragment labels (argmax), extract one SMILES per label.
    Returns (fragment_list, soft_cut_count).
    soft_cut_count = edges where p_keep < 0.5 (i.e. model voted to cut).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], 0

    labels = S_dense[0, :num_nodes, :].cpu().argmax(dim=-1).numpy()
    soft_cuts = int((p_keep.cpu() < 0.5).sum())

    frags: List[dict] = []
    for label in range(S_dense.shape[-1]):
        atom_idx = [i for i, l in enumerate(labels) if l == label]
        if not atom_idx:
            continue
        atom_set  = set(atom_idx)
        bond_idx  = [
            b.GetIdx() for b in mol.GetBonds()
            if b.GetBeginAtomIdx() in atom_set and b.GetEndAtomIdx() in atom_set
        ]
        try:
            smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_idx, bondsToUse=bond_idx)
        except Exception:
            smi = None
        if not smi:
            continue

        fmol       = Chem.MolFromSmiles(smi)
        is_valid   = False
        exact_mass = 0.0
        num_atoms  = len(atom_idx)
        if fmol is not None:
            try:
                Chem.SanitizeMol(fmol)
                is_valid   = True
                exact_mass = Descriptors.ExactMolWt(fmol)
                num_atoms  = fmol.GetNumAtoms()
            except Exception:
                pass

        frags.append(dict(
            smiles=smi, exact_mass=exact_mass,
            num_atoms=num_atoms, is_valid=is_valid,
        ))

    return frags, soft_cuts


def _peak_mzs(graph) -> List[float]:
    """Extract valid m/z values from a phase3_graphs_lpe.pt graph."""
    peaks = getattr(graph, "peaks", None)
    mask  = getattr(graph, "peak_mask", None)
    if peaks is None:
        return []
    if mask is not None:
        valid = peaks[~mask.bool()]
    else:
        valid = peaks
    # Column 0 is m/z divided by 1000 (preprocessing convention)
    if valid.shape[-1] >= 1:
        return (valid[:, 0] * 1000.0).tolist()
    return []


def _covered(fragment_masses: List[float], peak_mzs: List[float]) -> bool:
    """True if any fragment mass falls within 5 ppm of any experimental peak m/z."""
    for fm in fragment_masses:
        if fm <= 0:
            continue
        for pm in peak_mzs:
            if pm > 0 and abs(fm - pm) / pm < _PPM:
                return True
    return False


def _metrics(frags: List[dict], peak_mzs: List[float], avg_broken: float) -> dict:
    if not frags:
        return dict(n_frags=0, validity=0.0, peak_cov=0,
                    avg_atoms=0.0, unique=0, avg_broken=avg_broken)
    valid_frags = [f for f in frags if f["is_valid"]]
    masses      = [f["exact_mass"] for f in valid_frags]
    atoms       = [f["num_atoms"]  for f in frags]
    return dict(
        n_frags    = len(frags),
        validity   = len(valid_frags) / len(frags),
        peak_cov   = int(_covered(masses, peak_mzs)),
        avg_atoms  = float(np.mean(atoms)),
        unique     = len({f["smiles"] for f in frags if f["smiles"]}),
        avg_broken = avg_broken,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # ── Data loading ──────────────────────────────────────────────────────────
    print(f"\n[Load] spec_df   : {_PATHS['spec_df']}")
    spec_df = pd.read_pickle(_PATHS["spec_df"])
    spec_df["mol_id"] = spec_df["mol_id"].astype(int)

    print(f"[Load] mol_df    : {_PATHS['mol_df']}")
    mol_df = pd.read_pickle(_PATHS["mol_df"])
    mol_df["mol_id"] = mol_df["mol_id"].astype(int)
    mol_id_to_smiles = dict(zip(mol_df["mol_id"], mol_df["smiles"]))

    spec_to_mol: Dict[str, int] = dict(zip(spec_df["spec_id"], spec_df["mol_id"]))

    print(f"[Load] graphs    : {_PATHS['graphs']}")
    graphs = torch.load(_PATHS["graphs"], map_location="cpu",
                        weights_only=False)
    print(f"  → {len(graphs):,} graphs")

    # mol_id → [graph_index, …]
    mol_to_graphs: Dict[int, List[int]] = defaultdict(list)
    for gi, g in enumerate(graphs):
        sid = getattr(g, "spec_id", None)
        mid = spec_to_mol.get(sid)
        if mid is not None:
            mol_to_graphs[int(mid)].append(gi)

    print(f"[Load] ICEBERG cache: {_PATHS['ib_cache']}")
    ib_cache: Dict[int, List[dict]] = torch.load(
        _PATHS["ib_cache"], map_location="cpu", weights_only=False
    )
    ib_mol_ids: Set[int] = set(ib_cache.keys())
    print(f"  → {len(ib_mol_ids):,} mol_ids")

    # ── Intersect: molecules with BOTH cache entry AND a graph ────────────────
    shared = sorted(ib_mol_ids & set(mol_to_graphs.keys()))
    print(f"\n[Intersect] {len(shared):,} mol_ids have ICEBERG + Phase2.5 graph")

    rng = np.random.default_rng(args.seed)
    n   = min(args.num_samples, len(shared))
    sampled = sorted(rng.choice(shared, size=n, replace=False).tolist())
    print(f"[Sample]    {n} molecules (seed={args.seed})\n")

    # ── Load Phase 2.5 model ──────────────────────────────────────────────────
    print("[Model] Initialising DeterministicPhase2 …")
    p25 = DeterministicPhase2(_MODEL_CONFIG, max_fragments=10).to(device)
    ckpt_sd = torch.load(_PATHS["ckpt_p25"], map_location="cpu", weights_only=False)

    # Strip the "extractor." prefix so Siamese weights map to the base model
    extractor_sd = {
        k.replace("extractor.", ""): v
        for k, v in ckpt_sd.items()
        if k.startswith("extractor.")
    }

    # Fallback in case they loaded the old phase 2 weights by accident
    if len(extractor_sd) == 0:
        extractor_sd = ckpt_sd

    missing, unexpected = p25.load_state_dict(extractor_sd, strict=False)
    print(f"  → Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    p25.eval()

    # ── Evaluation loop ───────────────────────────────────────────────────────
    rows: List[dict] = []

    for mol_id in sampled:
        smiles = mol_id_to_smiles.get(mol_id)
        if not smiles:
            print(f"[{mol_id}] SKIP — no SMILES")
            continue

        # Peaks from the first available spectrum for this molecule
        gi     = mol_to_graphs[mol_id][0]
        pmzs   = _peak_mzs(graphs[gi])
        graph  = graphs[gi]

        print(f"[mol {mol_id}]  {smiles[:55]}  peaks={len(pmzs)}")

        # ─── ICEBERG ──────────────────────────────────────────────────────────
        ib_raw   = ib_cache.get(mol_id, [])[:args.top_k]
        ib_frags: List[dict] = []
        for fd in ib_raw:
            smi  = fd.get("smiles", "")
            if not smi:
                continue
            fmol = Chem.MolFromSmiles(smi)
            is_v = False
            mass = float(fd.get("exact_mass", 0.0))
            na   = fmol.GetNumAtoms() if fmol else 0
            if fmol is not None:
                try:
                    Chem.SanitizeMol(fmol)
                    is_v = True
                except Exception:
                    pass
            ib_frags.append(dict(
                smiles=smi, exact_mass=mass,
                num_atoms=na, is_valid=is_v,
                max_broken=int(fd.get("max_broken", 0)),
            ))

        ib_avg_broken = (
            float(np.mean([f["max_broken"] for f in ib_frags])) if ib_frags else 0.0
        )
        ib_m = _metrics(ib_frags, pmzs, ib_avg_broken)
        print(f"  ICEBERG  : n={ib_m['n_frags']:3d}  valid={ib_m['validity']:.2f}  "
              f"peak_cov={ib_m['peak_cov']}  avg_atoms={ib_m['avg_atoms']:.1f}  "
              f"unique={ib_m['unique']:3d}  avg_broken={ib_m['avg_broken']:.2f}")

        # ─── Phase 2.5 ────────────────────────────────────────────────────────
        pe = getattr(graph, "pe", None)
        p25_m = dict(n_frags=0, validity=float("nan"), peak_cov=float("nan"),
                     avg_atoms=float("nan"), unique=0, avg_broken=float("nan"))
        try:
            with torch.no_grad():
                batch = _build_p25_batch(smiles, device, pe=pe)
                if batch is None:
                    raise RuntimeError("batch construction failed")

                _, p_keep, S_dense, _ = p25(batch)
                num_nodes = int((batch["x"][0, :, 0] != 0).sum())
                p25_frags, soft_cuts = _p25_fragments(
                    smiles, S_dense, p_keep, num_nodes
                )

            p25_avg_broken = float(soft_cuts) / max(len(p25_frags), 1)
            p25_m = _metrics(p25_frags, pmzs, p25_avg_broken)
            print(f"  Phase2.5 : n={p25_m['n_frags']:3d}  valid={p25_m['validity']:.2f}  "
                  f"peak_cov={p25_m['peak_cov']}  avg_atoms={p25_m['avg_atoms']:.1f}  "
                  f"unique={p25_m['unique']:3d}  soft_cuts/frag={p25_m['avg_broken']:.2f}")

        except Exception:
            print("  Phase2.5 : FAILED")
            traceback.print_exc()

        rows.append({
            "mol_id": mol_id,
            "smiles_prefix": smiles[:50],
            "n_peaks":       len(pmzs),
            # ICEBERG columns
            "ib_n_frags":    ib_m["n_frags"],
            "ib_validity":   ib_m["validity"],
            "ib_peak_cov":   ib_m["peak_cov"],
            "ib_avg_atoms":  ib_m["avg_atoms"],
            "ib_unique":     ib_m["unique"],
            "ib_avg_broken": ib_m["avg_broken"],
            # Phase 2.5 columns
            "p25_n_frags":    p25_m["n_frags"],
            "p25_validity":   p25_m["validity"],
            "p25_peak_cov":   p25_m["peak_cov"],
            "p25_avg_atoms":  p25_m["avg_atoms"],
            "p25_unique":     p25_m["unique"],
            "p25_avg_broken": p25_m["avg_broken"],
        })

    # ── Aggregate summary ─────────────────────────────────────────────────────
    if not rows:
        print("\n[Summary] No results collected.")
        return

    df = pd.DataFrame(rows)

    metric_pairs = [
        ("Fragments / molecule",    "ib_n_frags",    "p25_n_frags"),
        ("Validity (%)",            "ib_validity",   "p25_validity"),
        ("Peak coverage (%)",       "ib_peak_cov",   "p25_peak_cov"),
        ("Avg fragment atom count", "ib_avg_atoms",  "p25_avg_atoms"),
        ("Unique SMILES / mol",     "ib_unique",     "p25_unique"),
        ("Avg bond-break count",    "ib_avg_broken", "p25_avg_broken"),
    ]

    print("\n" + "=" * 72)
    print("FRAGMENT ENGINE COMPARISON  (n={} molecules, seed={})".format(n, args.seed))
    print("=" * 72)

    summary = []
    for label, ib_col, p25_col in metric_pairs:
        ib_val  = df[ib_col].mean()
        p25_val = df[p25_col].mean()
        winner  = "ICEBERG" if ib_val > p25_val else ("Phase2.5" if p25_val > ib_val else "tie")
        # For validity and peak_cov, higher is better.
        # For avg_broken, lower is better (simpler fragments).
        if label == "Avg bond-break count":
            winner = "ICEBERG" if ib_val < p25_val else ("Phase2.5" if p25_val < ib_val else "tie")
        summary.append({
            "Metric":              label,
            "ICEBERG":             f"{ib_val:.3f}",
            "Phase 2.5":           f"{p25_val:.3f}",
            "Δ (P2.5 − IB)":       f"{p25_val - ib_val:+.3f}",
            "Better engine":       winner,
        })

    print(pd.DataFrame(summary).to_string(index=False))

    if args.save_csv:
        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\n[Save] per-molecule CSV → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare ICEBERG vs Phase 2.5 fragment engines on validation molecules."
    )
    p.add_argument("--num-samples", type=int, default=20,
                   help="Number of molecules to sample (default: 20)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--top-k", type=int, default=50,
                   help="Max ICEBERG fragments per molecule (default: 50)")
    p.add_argument("--device", default="cuda:0",
                   help="Torch device for Phase 2.5 inference (default: cuda:0)")
    p.add_argument("--save-csv", default=None,
                   help="Path to save per-molecule results as CSV")
    return p.parse_args()


if __name__ == "__main__":
    main(_cli())
