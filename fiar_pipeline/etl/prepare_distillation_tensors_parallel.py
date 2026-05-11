"""
Stage 2 Parallel Distillation Tensor Preparer
==============================================
Discovers all clean_spectra_*.parquet files in a directory and processes them
concurrently using multiprocessing.Pool, calling prepare_distillation_tensors
per file to merge MS3 arrays with SMILES metadata from the master index.

Replaces the Nextflow stage2_prepare_tensors.nf workflow for NRP Jobs where
Nextflow (requires Java) is unavailable.

Usage
-----
  python fiar_pipeline/etl/prepare_distillation_tensors_parallel.py \\
      --stage1_dir  /workspace/data/MSnLib/processed_stage1/ \\
      --master_index /workspace/data/MSnLib/master_metadata_index.parquet \\
      --outdir       /workspace/data/MSnLib/distillation_tensors/ \\
      --workers      8
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

from prepare_distillation_tensors import prepare_tensors


def _worker(task: tuple[str, str, str]) -> None:
    spectra_path, index_path, out_dir = task
    try:
        prepare_tensors(spectra_path, index_path, out_dir)
    except Exception as exc:
        print(f"[ERROR] {Path(spectra_path).name}: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel Stage 2 Distillation Tensor Preparer")
    parser.add_argument("--stage1_dir",    required=True, help="Directory containing clean_spectra_*.parquet files")
    parser.add_argument("--master_index",  required=True, help="Path to master_metadata_index.parquet")
    parser.add_argument("--outdir",        required=True, help="Output directory for distillation_tensors_*.parquet")
    parser.add_argument("--workers",       type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    spectra_files = sorted(Path(args.stage1_dir).glob("clean_spectra_*.parquet"))
    if not spectra_files:
        print(f"[!] No clean_spectra_*.parquet files found in {args.stage1_dir}")
        sys.exit(1)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print(f"Found {len(spectra_files)} stage-1 file(s). Launching {args.workers} worker(s)...",
          flush=True)

    tasks = [(str(f), args.master_index, args.outdir) for f in spectra_files]
    with Pool(processes=args.workers) as pool:
        pool.map(_worker, tasks)

    tensors = list(Path(args.outdir).glob("distillation_tensors_*.parquet"))
    total_rows = 0
    for t in tensors:
        import pandas as pd
        total_rows += len(pd.read_parquet(t, columns=["intact_parent_smiles"]))

    print(f"\nStage 2 complete — {len(tensors)} tensor file(s), {total_rows:,} total training rows written to {args.outdir}",
          flush=True)


if __name__ == "__main__":
    main()
