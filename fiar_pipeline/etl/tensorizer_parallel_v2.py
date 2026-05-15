"""
Stage 1 Parallel Tensorizer v2
================================
v2 of tensorizer_parallel.py — imports from tensorizer_v2 so each worker
emits the `rt_minutes` column needed by the Heuristic Fallback Matcher.

Usage
-----
  python fiar_pipeline/etl/tensorizer_parallel_v2.py \\
      --mzml_dir  /workspace/raw_scans/positive/ \\
      --outdir    /workspace/processed_stage1_v2/ \\
      --workers   8
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

from tensorizer_v2 import process_mzml_v2


def _worker(task: tuple[str, str]) -> None:
    mzml_path, out_dir = task
    try:
        process_mzml_v2(mzml_path, out_dir)
    except Exception as exc:
        print(f"[ERROR] {Path(mzml_path).name}: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel Stage 1 Tensorizer v2")
    parser.add_argument("--mzml_dir", required=True, help="Directory containing .mzML files")
    parser.add_argument("--outdir",   required=True, help="Output directory for .parquet files")
    parser.add_argument("--workers",  type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    mzml_files = sorted(Path(args.mzml_dir).glob("*.mzML"))
    if not mzml_files:
        print(f"[!] No .mzML files found in {args.mzml_dir}")
        sys.exit(1)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print(f"Found {len(mzml_files)} .mzML file(s). Launching {args.workers} worker(s)...",
          flush=True)

    tasks = [(str(f), args.outdir) for f in mzml_files]
    with Pool(processes=args.workers) as pool:
        pool.map(_worker, tasks)

    parquets = list(Path(args.outdir).glob("*.parquet"))
    print(f"\nStage 1 v2 complete — {len(parquets)} parquet file(s) written to {args.outdir}",
          flush=True)


if __name__ == "__main__":
    main()
