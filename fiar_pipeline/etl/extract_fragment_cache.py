"""
Offline ETL: ICEBERG Fragment Cache Builder
============================================
Runs ICEBERGScalpel over every unique molecule in mol_df and writes a
single .pt file (torch.save dict) that the Siamese DataLoader loads at
training time.

Output schema
-------------
fragment_cache.pt  →  Dict[mol_id: int, List[Dict]]

Per-fragment dict keys:
    smiles      str               canonical SMILES of the fragment
    exact_mass  float             monoisotopic mass
    prob_gen    float             autoregressive generation probability
    formula     str               molecular formula
    max_broken  int               cumulative bond-break count on path
    tree_depth  int               fragmentation depth
    morgan_fp   FloatTensor(2048) pre-computed Morgan fingerprint

Design decisions
----------------
- Format .pt  : variable-length per-mol lists don't fit rectangular formats
                (.feather / .parquet).  torch.load gives O(1) dict access.
- Resume flag : safe to restart after GPU OOM or timeout.
- Batch decode : FragGNN.predict_mol() accepts list inputs natively, giving
                 substantial GPU utilization over the ~28K molecule set.
- CE/adduct   : chosen per-molecule from spec_df using a priority ranking
                (see _representative_ce_adduct).

Usage
-----
cd /data/nas-gpu/wang/tmach007/ms-pred

/data/nas-gpu/wang/tmach007/ms-pred/y/envs/ms-gen/bin/python \\
    fiar_pipeline/etl/extract_fragment_cache.py \\
    --ckpt     weights/nist_iceberg_generate.ckpt \\
    --mol-df   /path/to/mol_df.pkl \\
    --spec-df  /path/to/spec_df.pkl \\
    --output   fiar_pipeline/data/fragment_cache.pt \\
    --device   cuda:0 \\
    --batch-size 32 \\
    --resume
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add repo root (for fiar_pipeline) and src/ (for ms_pred) to sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from fiar_pipeline.extractors.iceberg import ICEBERGScalpel


# ── Adduct / CE selection ─────────────────────────────────────────────────────

_ADDUCT_PRIORITY = {
    "[M+H]+": 0, "[M+Na]+": 1, "[M+NH4]+": 2,
    "[M-H]-": 3, "[M+Cl]-": 4,
}

_ICEBERG_ADDUCTS = {
    "[M+H]+", "[M+Na]+", "[M+NH4]+", "[M-H]-", "[M+Cl]-", "[M+K]+",
}


def _representative_ce_adduct(group: pd.DataFrame):
    """
    For a single molecule's spectrum rows, return (ce, adduct, prec_mz).

    Adduct: highest-priority ICEBERG-supported adduct by frequency.
    CE:     median nce_updated for that adduct; float('nan') if all missing
            (FragGNN handles NaN cleanly via collision_embed_merged fallback).
    """
    valid = group[group["prec_type"].isin(_ICEBERG_ADDUCTS)]
    if valid.empty:
        valid = group

    best_adduct = sorted(
        valid["prec_type"].value_counts().index.tolist(),
        key=lambda a: _ADDUCT_PRIORITY.get(a, 99),
    )[0]

    rows = valid[valid["prec_type"] == best_adduct]
    ce_vals = pd.to_numeric(rows["nce_updated"], errors="coerce").dropna()
    ce = float(ce_vals.median()) if len(ce_vals) else float("nan")
    prec_mz = float(rows["prec_mz"].iloc[0])
    return ce, best_adduct, prec_mz


# ── ETL core ──────────────────────────────────────────────────────────────────

def build_cache(args) -> dict:
    # ── Load metadata ─────────────────────────────────────────────────────────
    print("[ETL] Loading mol_df …")
    mol_df = pd.read_pickle(args.mol_df)
    print("[ETL] Loading spec_df …")
    spec_df = pd.read_pickle(args.spec_df)

    mol_smiles: dict[int, str] = dict(zip(mol_df["mol_id"].astype(int),
                                          mol_df["smiles"]))

    print("[ETL] Computing representative CE/adduct per molecule …")
    mol_meta: dict[int, dict] = {}
    for mol_id, group in tqdm(spec_df.groupby("mol_id"), desc="CE/Adduct"):
        mol_id = int(mol_id)
        smi = mol_smiles.get(mol_id)
        if not smi:
            continue
        ce, adduct, prec_mz = _representative_ce_adduct(group)
        mol_meta[mol_id] = dict(smiles=smi, ce=ce,
                                adduct=adduct, prec_mz=prec_mz)

    # ── Resume ────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache: dict = {}
    if args.resume and out_path.exists():
        cache = torch.load(out_path, weights_only=False)
        print(f"[ETL] Resuming — {len(cache)} mol_ids already cached.")

    todo = [mid for mid in mol_meta if mid not in cache]
    print(f"[ETL] Molecules to process: {len(todo)}")
    if not todo:
        print("[ETL] Nothing left to do.")
        return cache

    # ── Build scalpel ─────────────────────────────────────────────────────────
    scalpel = ICEBERGScalpel(
        ckpt_path=args.ckpt,
        device=args.device,
        top_k=args.top_k,
        threshold=args.threshold,
        compute_morgan_fp=True,
        morgan_radius=args.morgan_radius,
        morgan_nbits=args.morgan_nbits,
    )

    # ── Batched loop ──────────────────────────────────────────────────────────
    n_batches = math.ceil(len(todo) / args.batch_size)
    n_failed  = 0
    CKPT_EVERY = 500  # save checkpoint every N batches

    for b_idx in tqdm(range(n_batches), desc="Extracting"):
        batch_ids = todo[b_idx * args.batch_size:
                         (b_idx + 1) * args.batch_size]
        meta = [mol_meta[mid] for mid in batch_ids]

        try:
            all_frags = scalpel.extract_batch(
                smiles_list=[m["smiles"]  for m in meta],
                collision_engs=[m["ce"]      for m in meta],
                precursor_mzs=[m["prec_mz"] for m in meta],
                adducts=[m["adduct"]  for m in meta],
            )
        except Exception as exc:
            print(f"\n[ETL] Batch {b_idx} failed ({exc}). Skipping.")
            for mid in batch_ids:
                cache[mid] = []
            n_failed += len(batch_ids)
            continue

        for mid, frags in zip(batch_ids, all_frags):
            cache[mid] = [f.to_dict() for f in frags]

        if (b_idx + 1) % CKPT_EVERY == 0:
            torch.save(cache, out_path)
            print(f"\n[ETL] Checkpoint — {len(cache)} entries saved.")

    # ── Final save + report ───────────────────────────────────────────────────
    torch.save(cache, out_path)
    print(f"\n[ETL] Complete. Cache → {out_path}")
    print(f"      Total: {len(cache)} | Failed: {n_failed}")

    counts = [len(v) for v in cache.values() if v]
    if counts:
        print(f"      Avg frags/mol: {np.mean(counts):.1f} "
              f"(min={min(counts)}, max={max(counts)})")
    return cache


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(
        description="Build ICEBERG fragment cache (.pt)"
    )
    p.add_argument("--ckpt",          required=True,
                   help="FragGNN checkpoint, e.g. weights/nist_iceberg_generate.ckpt")
    p.add_argument("--mol-df",        required=True,
                   help="mol_df.pkl  (mol_id, smiles, ...)")
    p.add_argument("--spec-df",       required=True,
                   help="spec_df.pkl (mol_id, prec_type, nce_updated, prec_mz, ...)")
    p.add_argument("--output",        default="fiar_pipeline/data/fragment_cache.pt")
    p.add_argument("--top-k",         type=int,   default=100)
    p.add_argument("--threshold",     type=float, default=0.0)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--device",        default="cuda:0")
    p.add_argument("--morgan-radius", type=int,   default=2)
    p.add_argument("--morgan-nbits",  type=int,   default=2048)
    p.add_argument("--resume",        action="store_true",
                   help="Skip mol_ids already in --output")
    return p.parse_args()


if __name__ == "__main__":
    build_cache(_cli())
