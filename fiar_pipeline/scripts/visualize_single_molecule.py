"""
Diagnostic: visualize ICEBERG fragmentation for a single molecule.

Produces one image in fiar_pipeline/results/dag_diagnostics/:
  <prefix>_dag.png   — fragmentation DAG with 2D structures embedded in nodes

Run from repo root:
    /data/nas-gpu/wang/tmach007/ms-pred/y/envs/ms-gen/bin/python \\
        fiar_pipeline/scripts/visualize_single_molecule.py

Optional overrides (all have defaults):
    --smiles      "CC(=O)Oc1ccccc1C(=O)O"
    --adduct      "[M+H]+"
    --ce          40.0
    --prec-mz     181.05
    --ckpt        weights/nist_iceberg_generate.ckpt
    --device      cuda:0
    --top-k       50
    --output-dir  fiar_pipeline/results/dag_diagnostics
    --prefix      aspirin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Repo path setup ───────────────────────────────────────────────────────────
# Identical to extract_fragment_cache.py: add repo root (for fiar_pipeline)
# and src/ (for ms_pred) so this script runs from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from fiar_pipeline.extractors.iceberg import ICEBERGScalpel
from fiar_pipeline.extractors.iceberg.visualizer import ICEBERGDAGVisualizer


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    scalpel = ICEBERGScalpel(
        ckpt_path=args.ckpt,
        device=args.device,
        top_k=args.top_k,
        threshold=0.0,
        compute_morgan_fp=True,
    )

    # ── Run fragmentation ─────────────────────────────────────────────────────
    print(f"\n[Visualize] SMILES  : {args.smiles}")
    print(f"[Visualize] Adduct  : {args.adduct}  CE: {args.ce} eV  "
          f"Prec m/z: {args.prec_mz}")

    frag_hash_to_entry, fragments = scalpel.extract_with_dag(
        smiles=args.smiles,
        collision_eng=args.ce,
        precursor_mz=args.prec_mz,
        adduct=args.adduct,
    )

    print(f"[Visualize] DAG nodes   : {len(frag_hash_to_entry)}")
    print(f"[Visualize] Fragments   : {len(fragments)}")
    if fragments:
        depths = [f.tree_depth for f in fragments]
        print(f"[Visualize] Depth range : {min(depths)}–{max(depths)}")
        print(f"[Visualize] Top-5 by prob_gen:")
        for f in fragments[:5]:
            print(f"    {f.smiles:<40}  mass={f.exact_mass:.3f}  "
                  f"p={f.prob_gen:.4f}  d={f.tree_depth}  b={f.max_broken}")

    # ── Visualize ─────────────────────────────────────────────────────────────
    viz = ICEBERGDAGVisualizer(output_dir=str(output_dir))

    dag_path = viz.export_all(
        root_smiles=args.smiles,
        frag_hash_to_entry=frag_hash_to_entry,
        fragments=fragments,
        prefix=args.prefix,
    )

    print(f"\n[Visualize] Done.")
    print(f"  Embedded DAG → {dag_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize ICEBERG fragmentation DAG for a single molecule."
    )
    p.add_argument(
        "--smiles",     default="CC(=O)Oc1ccccc1C(=O)O",
        help="Input SMILES (default: aspirin)",
    )
    p.add_argument(
        "--adduct",     default="[M+H]+",
        help="Precursor adduct type (default: [M+H]+)",
    )
    p.add_argument(
        "--ce",         type=float, default=40.0,
        help="Collision energy in eV (default: 40.0)",
    )
    p.add_argument(
        "--prec-mz",    type=float, default=181.05,
        help="Precursor m/z (default: 181.05, i.e. aspirin [M+H]+)",
    )
    p.add_argument(
        "--ckpt",       default="weights/nist_iceberg_generate.ckpt",
        help="Path to FragGNN checkpoint",
    )
    p.add_argument(
        "--device",     default="cuda:0",
        help="Torch device (default: cuda:0)",
    )
    p.add_argument(
        "--top-k",      type=int, default=50,
        help="Max fragments to extract and display (default: 50)",
    )
    p.add_argument(
        "--output-dir", default="fiar_pipeline/results/dag_diagnostics",
        help="Directory for output PNGs",
    )
    p.add_argument(
        "--prefix",     default="aspirin",
        help="Filename prefix for output images (default: aspirin)",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(_cli())
