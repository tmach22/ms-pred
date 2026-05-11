import json
import glob
import pandas as pd
import sys
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

def build_master_index(json_dir, output_parquet):
    """
    Consolidates MSnLib JSON files into a single fast-lookup Parquet index.
    """
    json_files = glob.glob(f"{json_dir}/*.json")
    print(f"Found {len(json_files)} JSON files in {json_dir}. Building index...")

    if not json_files:
        print("Error: No JSON files found. Check the directory path.")
        sys.exit(1)

    records = []

    for file_path in json_files:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)

                    # Only index MS2-level records: stage1 scan_number = MS2 parent scan
                    if data.get("ms_level") != 2:
                        continue

                    # Handle cases where scan_number is a list (merged scans) vs single int
                    scans = data.get("scan_number", [])
                    if isinstance(scans, int):
                        scans = [scans]

                    # Prefer raw_file_name (targetmol format); fall back to feature_id (older format)
                    raw_file = data.get("raw_file_name", "")
                    feature_id = str(data.get("feature_id", ""))
                    if raw_file:
                        mzml_file = raw_file
                    elif ".mzML" in feature_id:
                        mzml_file = feature_id.split(".mzML")[0] + ".mzML"
                    else:
                        continue  # skip unresolvable records

                    for scan in scans:
                        records.append({
                            "mzml_file": mzml_file,
                            "scan_number": scan,
                            "smiles": data.get("smiles", ""),
                            "ms1_precursor_mz": data.get("precursor_mz", 0.0),
                            "collision_energy": str(data.get("collision_energy", ""))
                        })
                except json.JSONDecodeError:
                    continue

    # Convert to DataFrame and drop duplicates
    df = pd.DataFrame(records)
    initial_len = len(df)
    df = df.drop_duplicates(subset=["mzml_file", "scan_number"])
    final_len = len(df)

    print(f"Processed {initial_len} total scan references. Deduplicated to {final_len} unique scans.")

    # Save as Parquet
    df.to_parquet(output_parquet, index=False, engine='pyarrow')
    print(f"Master index built successfully! Saved to: {output_parquet}")
    print(f"File size: {os.path.getsize(output_parquet) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    json_directory = _REPO_ROOT / "data" / "MSnLib" / "libraries" / "json"
    output_file    = _REPO_ROOT / "data" / "MSnLib" / "master_metadata_index.parquet"

    build_master_index(str(json_directory), str(output_file))
