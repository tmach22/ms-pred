"""
Stage 1 Tensorizer
==================
Parses a raw .mzML file, extracts all MS3 scans and their MS2 parent
reference, applies the ICEBERG spectral transformations, and writes a
lightweight .parquet file containing the cleaned spectral tensors.

Transformations applied (in order)
------------------------------------
1. Impossibility pruning  — drop peaks with m/z > MS2 precursor + 1.5 Da
2. Square-root normalisation — int_sqrt = sqrt(I / I_max)
3. Top-50 thresholding    — keep the 50 highest-intensity peaks

Output schema (one row per MS3 scan)
--------------------------------------
  scan_number       int    — parent MS2 scan number (from spectrum reference)
  ms2_precursor_mz  float  — m/z of the MS2 precursor that was re-fragmented
  ms3_mz            list   — pruned/thresholded m/z array
  ms3_intensity     list   — sqrt-normalised intensity array

Usage
-----
  python fiar_pipeline/etl/tensorizer.py \\
      --mzml  <path/to/file.mzML> \\
      --outdir <output_directory>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as ms


# ── Transformation ────────────────────────────────────────────────────────────

def transform_ms3_peaks(
    raw_mz: np.ndarray,
    raw_int: np.ndarray,
    ms2_precursor_mz: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Apply impossibility pruning, sqrt normalisation, and Top-50 threshold."""
    valid = raw_mz <= (ms2_precursor_mz + 1.5)
    mz_pruned  = raw_mz[valid]
    int_pruned = raw_int[valid]

    if len(int_pruned) == 0:
        return None, None

    int_sqrt = np.sqrt(int_pruned / np.max(int_pruned))

    if len(mz_pruned) > 50:
        top50 = np.argsort(int_sqrt)[-50:]
        return mz_pruned[top50], int_sqrt[top50]

    return mz_pruned, int_sqrt


# ── File-level processor ──────────────────────────────────────────────────────

def process_mzml(mzml_path: str, out_dir: str) -> None:
    print(f"Parsing {mzml_path}...")
    exp = ms.MSExperiment()
    ms.MzMLFile().load(mzml_path, exp)

    records: list[dict] = []

    for spec in exp:
        if spec.getMSLevel() != 3:
            continue

        precursors = spec.getPrecursors()
        if not precursors:
            continue

        ms2_mz = precursors[0].getMZ()

        # Resolve parent MS2 scan number from spectrum reference metadata.
        # pyopenms exposes this as 'spectrum_ref' (underscore, bytes key).
        # Fallback: parse the scan number from the spectrum's own native ID.
        parent_ref = precursors[0].getMetaValue("spectrum_ref")
        try:
            if parent_ref:
                ref_str = parent_ref.decode("utf-8") if isinstance(parent_ref, bytes) else str(parent_ref)
                parent_scan_num = int(ref_str.split("=")[-1])
            else:
                native_id = spec.getNativeID()
                if isinstance(native_id, bytes):
                    native_id = native_id.decode("utf-8")
                parent_scan_num = int(native_id.split("scan=")[-1])
        except (ValueError, AttributeError):
            continue

        raw_mz, raw_int = spec.get_peaks()
        final_mz, final_int = transform_ms3_peaks(raw_mz, raw_int, ms2_mz)

        if final_mz is not None and len(final_mz) > 0:
            records.append({
                "scan_number":      parent_scan_num,
                "ms2_precursor_mz": float(ms2_mz),
                "ms3_mz":           final_mz.tolist(),
                "ms3_intensity":    final_int.tolist(),
            })

    if not records:
        print(f"No valid MS3 spectra extracted from {mzml_path}")
        return

    df = pd.DataFrame(records)
    base_name = Path(mzml_path).stem
    out_path  = Path(out_dir) / f"clean_spectra_{base_name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")

    print(f"SUCCESS: Saved {len(records)} transformed MS3 spectra to {out_path}")
    print(f"Dataframe shape: {df.shape}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 Tensorizer — mzML → Parquet")
    parser.add_argument("--mzml",   required=True, help="Path to raw .mzML file")
    parser.add_argument("--outdir", required=True, help="Directory for output .parquet")
    args = parser.parse_args()

    process_mzml(args.mzml, args.outdir)
