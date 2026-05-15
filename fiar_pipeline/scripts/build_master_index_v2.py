"""
build_master_index_v2.py
========================
v2 of build_master_index.py — adds `rt` (float, minutes) and `precursor_mz`
(float, MS2 precursor m/z) to the Parquet schema so that Stage 2 v2's
Heuristic Fallback Matcher can use physical anchors for Tier 2 matching.

Changes vs v1:
  - `precursor_mz` column added (replaces the renamed `ms1_precursor_mz`)
  - `rt`           column added (float, already in minutes in MSnLib JSONs)
  - Output written to master_metadata_index_v2.parquet

Output schema:
  mzml_file       str    — source mzML filename
  scan_number     int    — MS2 scan number (one row per scan)
  smiles          str    — canonical SMILES for the parent molecule
  precursor_mz    float  — MS2 precursor m/z ([M+H]+ or [M-H]−)
  rt              float  — retention time of the MS2 scan (minutes)
  collision_energy str   — list of collision energies used
"""

import json
import glob
import os
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]


def build_master_index_v2(json_dir: str, output_parquet: str) -> None:
    json_files = sorted(glob.glob(f"{json_dir}/*.json"))
    print(f"Found {len(json_files)} JSON files in {json_dir}. Building v2 index...")

    if not json_files:
        print("Error: No JSON files found. Check the directory path.")
        sys.exit(1)

    records = []

    for file_path in json_files:
        with open(file_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)

                    if data.get("ms_level") != 2:
                        continue

                    # scan_number can be a list (merged scans) or a single int
                    scans = data.get("scan_number", [])
                    if isinstance(scans, int):
                        scans = [scans]

                    # Resolve source mzML filename
                    raw_file   = data.get("raw_file_name", "")
                    feature_id = str(data.get("feature_id", ""))
                    if raw_file:
                        mzml_file = raw_file
                    elif ".mzML" in feature_id:
                        mzml_file = feature_id.split(".mzML")[0] + ".mzML"
                    else:
                        continue

                    # v2: extract rt and precursor_mz explicitly
                    rt_val          = float(data.get("rt", 0.0) or 0.0)
                    precursor_mz_val = float(data.get("precursor_mz", 0.0) or 0.0)

                    for scan in scans:
                        records.append({
                            "mzml_file":      mzml_file,
                            "scan_number":    int(scan),
                            "smiles":         data.get("smiles", ""),
                            "precursor_mz":   precursor_mz_val,
                            "rt":             rt_val,
                            "collision_energy": str(data.get("collision_energy", "")),
                        })

                except (json.JSONDecodeError, TypeError, ValueError):
                    continue

    df = pd.DataFrame(records)
    initial_len = len(df)
    df = df.drop_duplicates(subset=["mzml_file", "scan_number"])
    final_len = len(df)

    print(f"Processed {initial_len} total scan references. "
          f"Deduplicated to {final_len} unique (mzml_file, scan_number) pairs.")

    # Ensure correct dtypes
    df["scan_number"]  = df["scan_number"].astype("int64")
    df["precursor_mz"] = df["precursor_mz"].astype("float64")
    df["rt"]           = df["rt"].astype("float64")

    df.to_parquet(output_parquet, index=False, engine="pyarrow")
    size_mb = os.path.getsize(output_parquet) / (1024 * 1024)
    print(f"v2 master index saved to: {output_parquet}  ({size_mb:.2f} MB)")
    print(f"Schema: {df.columns.tolist()}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    json_directory = _REPO_ROOT / "data" / "MSnLib" / "libraries" / "json"
    output_file    = _REPO_ROOT / "data" / "MSnLib" / "master_metadata_index_v2.parquet"

    build_master_index_v2(str(json_directory), str(output_file))
