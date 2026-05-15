"""
Stage 1 Tensorizer v2
=====================
v2 of tensorizer.py — adds `rt_minutes` (float) to the output schema so that
Stage 2 v2's Heuristic Fallback Matcher can use RT as a physical anchor for
Tier 2 matching when exact scan_number joins fail.

Changes vs v1:
  - Pre-builds an MS2 scan_number → RT (minutes) lookup from the same mzML
  - Each output row gains `rt_minutes` (parent MS2 scan RT, seconds / 60)
  - Main entry point renamed to `process_mzml_v2`; v1 signature preserved as
    `process_mzml` for backwards-compatible imports.

Output schema (one row per MS3 scan)
--------------------------------------
  scan_number       int    — parent MS2 scan number
  ms2_precursor_mz  float  — m/z of the MS2 precursor re-fragmented to MS3
  rt_minutes        float  — retention time of the parent MS2 scan (minutes)
  ms3_mz            list   — pruned/thresholded m/z array
  ms3_intensity     list   — sqrt-normalised intensity array

Usage
-----
  python fiar_pipeline/etl/tensorizer_v2.py \\
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


# ── Transformation (unchanged from v1) ────────────────────────────────────────

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


# ── v2 File-level processor ───────────────────────────────────────────────────

def process_mzml_v2(mzml_path: str, out_dir: str) -> None:
    print(f"[v2] Parsing {mzml_path}...")
    exp = ms.MSExperiment()
    ms.MzMLFile().load(mzml_path, exp)

    # Pass 1: build RT lookup from every MS2 scan
    ms2_rt_lookup: dict[int, float] = {}
    for spec in exp:
        if spec.getMSLevel() != 2:
            continue
        native_id = spec.getNativeID()
        if isinstance(native_id, bytes):
            native_id = native_id.decode("utf-8")
        try:
            scan_num = int(native_id.split("scan=")[-1])
            ms2_rt_lookup[scan_num] = spec.getRT() / 60.0  # seconds → minutes
        except (ValueError, AttributeError):
            pass

    print(f"[v2] Built RT lookup for {len(ms2_rt_lookup)} MS2 scans.")

    # Pass 2: extract MS3 scans with parent RT
    records: list[dict] = []

    for spec in exp:
        if spec.getMSLevel() != 3:
            continue

        precursors = spec.getPrecursors()
        if not precursors:
            continue

        ms2_mz = precursors[0].getMZ()

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

        rt_minutes = ms2_rt_lookup.get(parent_scan_num, 0.0)

        raw_mz, raw_int = spec.get_peaks()
        final_mz, final_int = transform_ms3_peaks(raw_mz, raw_int, ms2_mz)

        if final_mz is not None and len(final_mz) > 0:
            records.append({
                "scan_number":      parent_scan_num,
                "ms2_precursor_mz": float(ms2_mz),
                "rt_minutes":       rt_minutes,
                "ms3_mz":           final_mz.tolist(),
                "ms3_intensity":    final_int.tolist(),
            })

    if not records:
        print(f"[v2] No valid MS3 spectra extracted from {mzml_path}")
        return

    df = pd.DataFrame(records)
    base_name = Path(mzml_path).stem
    out_path  = Path(out_dir) / f"clean_spectra_{base_name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")

    print(f"[v2] SUCCESS: Saved {len(records)} transformed MS3 spectra to {out_path}")
    print(f"[v2] Dataframe shape: {df.shape}")


# Backwards-compatible alias so v1 parallel scripts still work if imported
process_mzml = process_mzml_v2


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 Tensorizer v2 — mzML → Parquet")
    parser.add_argument("--mzml",   required=True, help="Path to raw .mzML file")
    parser.add_argument("--outdir", required=True, help="Directory for output .parquet")
    args = parser.parse_args()

    process_mzml_v2(args.mzml, args.outdir)
