"""
Dataset-Sample DAG Diagnostic
==============================
Randomly samples N molecules from mol_df + spec_df, derives each molecule's
representative collision energy and adduct using the same logic as the ETL,
then runs ICEBERGScalpel.extract_with_dag() and exports an embedded DAG image
with cleavage-site highlighting for each molecule.

Run from repo root:
    /data/nas-gpu/wang/tmach007/ms-pred/y/envs/ms-gen/bin/python \\
        fiar_pipeline/scripts/visualize_dataset_samples.py

All arguments have sensible defaults pointing at the COMBINED datasets.
See --help for the full list.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

# ── Repo path setup ───────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from fiar_pipeline.extractors.iceberg import ICEBERGScalpel
from fiar_pipeline.extractors.iceberg.visualizer import ICEBERGDAGVisualizer
from fiar_pipeline.etl.extract_fragment_cache import (
    _representative_ce_adduct,
    _ICEBERG_ADDUCTS,
)

# Proton mass used to approximate [M+H]+ prec_mz when adduct fallback is needed
_PROTON = 1.007276


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_molecules(
    mol_df: pd.DataFrame,
    spec_df: pd.DataFrame,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """
    Return a DataFrame of `n` randomly sampled rows from mol_df where the
    molecule has at least one spectrum in spec_df (required for CE/adduct
    selection).  Sampling is reproducible via `seed`.
    """
    mol_ids_with_spectra = set(spec_df["mol_id"].unique())
    eligible = mol_df[mol_df["mol_id"].isin(mol_ids_with_spectra)].copy()
    n_eligible = len(eligible)
    if n > n_eligible:
        print(f"[Sample] Warning: requested {n} but only {n_eligible} "
              "molecules have spectra — using all.")
        n = n_eligible
    return eligible.sample(n=n, random_state=seed).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"[Sample] Loading mol_df  : {args.mol_df}")
    mol_df = pd.read_pickle(args.mol_df)
    mol_df["mol_id"] = mol_df["mol_id"].astype(int)

    print(f"[Sample] Loading spec_df : {args.spec_df}")
    spec_df = pd.read_pickle(args.spec_df)
    spec_df["mol_id"] = spec_df["mol_id"].astype(int)

    print(f"[Sample] mol_df: {len(mol_df):,} molecules | "
          f"spec_df: {len(spec_df):,} spectra")

    # ── Sample ────────────────────────────────────────────────────────────────
    sample = _sample_molecules(mol_df, spec_df, args.num_samples, args.seed)
    print(f"\n[Sample] Selected {len(sample)} molecules "
          f"(seed={args.seed}):\n")
    for _, row in sample.iterrows():
        print(f"  mol_id={row['mol_id']:>8}  {row['smiles'][:70]}")
    print()

    # ── Build spec lookup: mol_id → group of spectrum rows ────────────────────
    spec_grouped = {
        int(mid): grp
        for mid, grp in spec_df.groupby("mol_id")
    }

    # ── Load model + visualizer ───────────────────────────────────────────────
    scalpel = ICEBERGScalpel(
        ckpt_path=args.ckpt,
        device=args.device,
        top_k=args.top_k,
        threshold=0.0,
        compute_morgan_fp=False,    # not needed for diagnostics
    )
    viz = ICEBERGDAGVisualizer(output_dir=str(output_dir))

    # ── Per-molecule loop ─────────────────────────────────────────────────────
    succeeded, failed = 0, 0

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        mol_id = int(row["mol_id"])
        smiles = row["smiles"]
        prefix = f"mol_{mol_id}"

        print(f"[{i}/{len(sample)}] mol_id={mol_id}  smiles={smiles[:60]}")

        try:
            # Derive CE / adduct / prec_mz using the same ETL logic
            group = spec_grouped.get(mol_id)
            if group is None or group.empty:
                print(f"  → SKIP: no spectra found for mol_id={mol_id}")
                failed += 1
                continue

            ce, adduct, prec_mz = _representative_ce_adduct(group)

            # The COMBINED dataset contains adducts outside FragGNN's vocabulary
            # (e.g. [2M+NA]+).  Fall back to [M+H]+ using exact_mw from mol_df.
            if adduct not in _ICEBERG_ADDUCTS:
                orig = adduct
                adduct = "[M+H]+"
                ce = float("nan")
                exact_mw = row.get("exact_mw")
                if exact_mw is not None and not pd.isna(exact_mw):
                    prec_mz = float(exact_mw) + _PROTON
                print(f"  Warning: adduct '{orig}' not in model vocab → "
                      f"fell back to [M+H]+")

            print(f"  adduct={adduct}  CE={ce:.1f} eV  prec_mz={prec_mz:.4f}")

            # Extract DAG
            frag_hash_to_entry, fragments = scalpel.extract_with_dag(
                smiles=smiles,
                collision_eng=ce,
                precursor_mz=prec_mz,
                adduct=adduct,
            )
            print(f"  DAG nodes={len(frag_hash_to_entry)}  "
                  f"fragments={len(fragments)}  "
                  f"depth={max((f.tree_depth for f in fragments), default=0)}")

            # Export embedded DAG
            dag_path = viz.export_all(
                root_smiles=smiles,
                frag_hash_to_entry=frag_hash_to_entry,
                fragments=fragments,
                prefix=prefix,
            )
            print(f"  → {dag_path}")
            succeeded += 1

        except Exception:
            print(f"  → FAILED (mol_id={mol_id}):")
            traceback.print_exc()
            failed += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[Sample] Done.  {succeeded} succeeded, {failed} failed.")
    print(f"[Sample] Images → {output_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> argparse.Namespace:
    _SSP = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor"
    p = argparse.ArgumentParser(
        description="Visualize ICEBERG DAGs for randomly sampled dataset molecules."
    )
    p.add_argument(
        "--mol-df",
        default=f"{_SSP}/mass_spec_gym_data/mol_df_COMBINED.pkl",
        help="Path to mol_df pickle (default: mol_df_COMBINED.pkl)",
    )
    p.add_argument(
        "--spec-df",
        default=f"{_SSP}/mass_spec_gym_data/spec_df_COMBINED.pkl",
        help="Path to spec_df pickle (default: spec_df_COMBINED.pkl)",
    )
    p.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of molecules to sample (default: 5)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    p.add_argument(
        "--ckpt", default="weights/nist_iceberg_generate.ckpt",
        help="Path to FragGNN checkpoint",
    )
    p.add_argument(
        "--device", default="cuda:0",
        help="Torch device (default: cuda:0)",
    )
    p.add_argument(
        "--top-k", type=int, default=50,
        help="Max fragments to extract per molecule (default: 50)",
    )
    p.add_argument(
        "--output-dir", default="fiar_pipeline/results/dag_diagnostics",
        help="Directory for output PNGs",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(_cli())
