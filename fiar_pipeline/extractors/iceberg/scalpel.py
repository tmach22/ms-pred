"""
ICEBERGScalpel
==============
Lives inside ms-pred, so ms_pred.* imports are unconditional — no sys.path
manipulation required.  This is the key architectural advantage of the
inside-repo containment strategy.

Returns Top-K explicit 2D fragment records (SMILES, exact mass, prob_gen,
pre-computed Morgan FP) from the autoregressive FragGNN DAG.

Usage
-----
from fiar_pipeline.extractors.iceberg import ICEBERGScalpel

scalpel = ICEBERGScalpel.from_config(cfg["iceberg"])
frags   = scalpel.extract("CC(=O)Oc1ccccc1C(=O)O",
                           collision_eng=40.0, precursor_mz=181.05,
                           adduct="[M+H]+")
# or batched:
all_frags = scalpel.extract_batch(smiles_list, ces, mzs, adducts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

# Direct ms_pred imports — no path tricks needed inside the repo
from ms_pred.dag_pred.gen_model import FragGNN
from ms_pred.magma.fragmentation import FragmentEngine


# ── SMILES recovery ───────────────────────────────────────────────────────────

def _frag_int_to_smiles(engine: FragmentEngine, frag_int: int) -> Optional[str]:
    """
    Converts an integer bitmask fragment back to a SMILES string.

    FragmentEngine.get_draw_dict() returns the parent Mol with atom/bond
    index sets for the fragment.  Chem.MolFragmentToSmiles() converts these
    directly to a valid, canonical SMILES fragment.

    This is the only correct recovery path — FragmentEngine has no
    get_frag_smiles() method.
    """
    from rdkit import Chem

    try:
        d = engine.get_draw_dict(frag_int)
        smi = Chem.MolFragmentToSmiles(
            d["mol"],
            atomsToUse=list(d["hatoms"]),
            bondsToUse=d["hbonds"],
        )
        return smi or None
    except Exception:
        return None


# ── Data record ───────────────────────────────────────────────────────────────

@dataclass
class FragmentRecord:
    """One decoded fragment from the ICEBERG-Generate DAG."""
    smiles:      str
    exact_mass:  float
    prob_gen:    float
    formula:     str
    max_broken:  int
    tree_depth:  int
    morgan_fp:            Optional[torch.FloatTensor] = field(default=None, repr=False)
    # Visualization-only fields — not serialised by to_dict(), not compared in __eq__
    root_mol:             Optional[object]    = field(default=None, compare=False, repr=False)
    frag_atom_indices:    Optional[frozenset] = field(default=None, compare=False, repr=False)
    frag_bond_indices:    Optional[list]      = field(default=None, compare=False, repr=False)
    cleavage_atom_indices: Optional[list]     = field(default=None, compare=False, repr=False)

    def to_dict(self) -> dict:
        d = dict(
            smiles=self.smiles,
            exact_mass=self.exact_mass,
            prob_gen=self.prob_gen,
            formula=self.formula,
            max_broken=self.max_broken,
            tree_depth=self.tree_depth,
        )
        if self.morgan_fp is not None:
            d["morgan_fp"] = self.morgan_fp
        return d


# ── Main class ────────────────────────────────────────────────────────────────

class ICEBERGScalpel:
    """
    Wraps a frozen FragGNN checkpoint and exposes fragment extraction for
    both single-molecule and batched workflows.

    Parameters
    ----------
    ckpt_path         : path to nist_iceberg_generate.ckpt (or canopus variant)
    device            : torch device string, e.g. "cuda:0" or "cpu"
    top_k             : max fragments per molecule
    threshold         : minimum atom-leaving probability (0.0 = keep all)
    compute_morgan_fp : if True, pre-computes 2048-bit Morgan FP for each fragment
    morgan_radius     : Morgan radius (default 2)
    morgan_nbits      : fingerprint bit length (default 2048)
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        top_k: int = 100,
        threshold: float = 0.0,
        compute_morgan_fp: bool = True,
        morgan_radius: int = 2,
        morgan_nbits: int = 2048,
    ):
        self.device = device
        self.top_k = top_k
        self.threshold = threshold
        self.compute_morgan_fp = compute_morgan_fp
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits

        print(f"[ICEBERGScalpel] Loading: {ckpt_path}")
        self.model: FragGNN = FragGNN.load_from_checkpoint(
            ckpt_path, map_location=device
        )
        self.model.eval()
        self.model.freeze()
        self.model.to(device)
        print(f"[ICEBERGScalpel] Ready on {device}.")

    @classmethod
    def from_config(cls, cfg: dict) -> "ICEBERGScalpel":
        """Construct from the 'iceberg' section of a YAML config dict."""
        return cls(
            ckpt_path=cfg["ckpt_path"],
            device=cfg.get("device", "cpu"),
            top_k=cfg.get("top_k", 100),
            threshold=cfg.get("threshold", 0.0),
            compute_morgan_fp=cfg.get("compute_morgan_fp", True),
            morgan_radius=cfg.get("morgan_radius", 2),
            morgan_nbits=cfg.get("morgan_nbits", 2048),
        )

    # ── Internal decode ───────────────────────────────────────────────────────

    def _decode(
        self, smiles: str, frag_hash_to_entry: dict
    ) -> List[FragmentRecord]:
        """
        Decode one molecule's frag_hash_to_entry dict into FragmentRecords.

        Two-pass approach:
          Pass 1 — pre-compute frag_atom_indices for every entry so that each
                   fragment can resolve its immediate parent's atom set without
                   a second FragmentEngine call.
          Pass 2 — build FragmentRecord, computing cleavage_atom_indices as the
                   atoms in this fragment that were directly bonded to atoms
                   removed by the MOST RECENT cut (relative to immediate parent).
        """
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        engine = FragmentEngine(mol_str=smiles)

        # ── Pass 1: map every hash → frozenset of root-mol atom indices ───────
        hash_to_frag_atoms: dict = {}
        for frag_hash, entry in frag_hash_to_entry.items():
            try:
                d = engine.get_draw_dict(entry["frag"])
                hash_to_frag_atoms[frag_hash] = frozenset(d["hatoms"])
            except Exception:
                hash_to_frag_atoms[frag_hash] = None

        # ── Pass 2: decode each entry ─────────────────────────────────────────
        records: List[FragmentRecord] = []

        for frag_hash, entry in frag_hash_to_entry.items():
            frag_smi = _frag_int_to_smiles(engine, entry["frag"])
            if frag_smi is None:
                continue
            smi_mol = Chem.MolFromSmiles(frag_smi)
            if smi_mol is None:
                continue

            exact_mass = Descriptors.ExactMolWt(smi_mol)

            # ── Draw-dict fields (visualization only) ─────────────────────────
            root_mol = None
            frag_atom_indices: Optional[frozenset] = None
            frag_bond_indices: Optional[list] = None
            cleavage_atom_indices: Optional[list] = None

            try:
                d = engine.get_draw_dict(entry["frag"])
                root_mol         = d["mol"]
                frag_atom_indices = frozenset(d["hatoms"])
                frag_bond_indices = list(d["hbonds"])

                # Find the best immediate parent: largest frozenset that is a
                # strict superset of this fragment's atom set.
                best_parent: Optional[frozenset] = None
                for parent_hash in entry.get("parents", []):
                    pa = hash_to_frag_atoms.get(parent_hash)
                    if (pa is not None
                            and pa.issuperset(frag_atom_indices)
                            and pa != frag_atom_indices
                            and (best_parent is None or len(pa) > len(best_parent))):
                        best_parent = pa

                if best_parent is not None:
                    just_removed = best_parent - frag_atom_indices
                    cleavage_atom_indices = [
                        idx for idx in frag_atom_indices
                        if any(
                            nbr.GetIdx() in just_removed
                            for nbr in root_mol.GetAtomWithIdx(idx).GetNeighbors()
                        )
                    ]
                else:
                    cleavage_atom_indices = []  # root fragment — no cut to show

            except Exception:
                pass

            # ── Morgan fingerprint ────────────────────────────────────────────
            morgan_fp = None
            if self.compute_morgan_fp:
                bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    smi_mol, self.morgan_radius, nBits=self.morgan_nbits
                )
                morgan_fp = torch.tensor(
                    [int(b) for b in bv.ToBitString()], dtype=torch.float32
                )

            records.append(FragmentRecord(
                smiles=frag_smi,
                exact_mass=exact_mass,
                prob_gen=entry.get("prob_gen", 0.0),
                formula=entry.get("form", ""),
                max_broken=entry.get("max_broken", 0),
                tree_depth=entry.get("tree_depth", 0),
                morgan_fp=morgan_fp,
                root_mol=root_mol,
                frag_atom_indices=frag_atom_indices,
                frag_bond_indices=frag_bond_indices,
                cleavage_atom_indices=cleavage_atom_indices,
            ))

        records.sort(key=lambda r: -r.prob_gen)
        return records[: self.top_k]

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(
        self,
        smiles: str,
        collision_eng: float,
        precursor_mz: float,
        adduct: str = "[M+H]+",
        instrument: Optional[str] = None,
    ) -> List[FragmentRecord]:
        """Run FragGNN on a single SMILES and return top-K fragments."""
        result = self.model.predict_mol(
            root_smi=smiles,
            collision_eng=collision_eng,
            precursor_mz=precursor_mz,
            adduct=adduct,
            instrument=instrument,
            threshold=self.threshold,
            device=self.device,
            max_nodes=self.top_k,
            canonical_root_smi=False,
        )
        return self._decode(smiles, result)

    def extract_with_dag(
        self,
        smiles: str,
        collision_eng: float,
        precursor_mz: float,
        adduct: str = "[M+H]+",
        instrument: Optional[str] = None,
    ):
        """
        Like extract(), but also returns the raw frag_hash_to_entry dict needed
        by ICEBERGDAGVisualizer.plot_dag().

        Returns
        -------
        (frag_hash_to_entry: dict, fragments: List[FragmentRecord])
        """
        frag_hash_to_entry = self.model.predict_mol(
            root_smi=smiles,
            collision_eng=collision_eng,
            precursor_mz=precursor_mz,
            adduct=adduct,
            instrument=instrument,
            threshold=self.threshold,
            device=self.device,
            max_nodes=self.top_k,
            canonical_root_smi=False,
        )
        fragments = self._decode(smiles, frag_hash_to_entry)
        return frag_hash_to_entry, fragments

    def extract_batch(
        self,
        smiles_list: List[str],
        collision_engs: List[float],
        precursor_mzs: List[float],
        adducts: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None,
    ) -> List[List[FragmentRecord]]:
        """
        Run FragGNN in batched mode.  FragGNN.predict_mol() natively accepts
        list inputs, so this is the GPU-efficient path for the ETL.
        """
        if adducts is None:
            adducts = ["[M+H]+"] * len(smiles_list)
        if instruments is None:
            instruments = [None] * len(smiles_list)

        batch_results = self.model.predict_mol(
            root_smi=smiles_list,
            collision_eng=collision_engs,
            precursor_mz=precursor_mzs,
            adduct=adducts,
            instrument=instruments,
            threshold=self.threshold,
            device=self.device,
            max_nodes=self.top_k,
            canonical_root_smi=False,
        )

        return [
            self._decode(smi, entry_map)
            for smi, entry_map in zip(smiles_list, batch_results)
        ]
