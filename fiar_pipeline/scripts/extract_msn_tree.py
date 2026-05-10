"""
MSn Tree Extractor (JSON edition)
==================================
Parses a MSnLib NDJSON file to verify the hierarchical linkage of
MS2 and MS3 spectra, mirroring what the pyopenms/mzML version would show.

Schema used
-----------
  ms_level            : 2 or 3
  precursor_mz        : float  — MS1 precursor m/z
  msn_precursor_mzs   : list
      MS2 entry  → [ms2_prec_mz]
      MS3 entry  → [ms2_prec_mz, ms3_prec_mz]
  feature_id          : str    — encodes source mzML file and scan numbers
  num_peaks / peaks   : int / list

Usage
-----
  python extract_msn_tree.py <path_to_*_msn.json>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def extract_msn_tree(json_path: str, max_show: int = 5) -> None:
    path = Path(json_path)
    ms2_entries = []  # ms_level == 2
    ms3_entries = []  # ms_level == 3

    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                e = json.loads(raw)
            except json.JSONDecodeError:
                print(f"  [WARN] line {lineno}: JSON parse error", file=sys.stderr)
                continue

            lvl = e.get("ms_level")
            if lvl == 2:
                ms2_entries.append(e)
            elif lvl == 3:
                ms3_entries.append(e)

    total = len(ms2_entries) + len(ms3_entries)
    print(f"Loaded {total} spectra from {path.name}")
    print(f"  MS2 spectra : {len(ms2_entries):,}")
    print(f"  MS3 spectra : {len(ms3_entries):,}")

    if not ms3_entries:
        print("No MS3 linkages found in this file.")
        return

    # ── Build hierarchy ────────────────────────────────────────────────────────
    # Key: rounded(MS2_precursor_mz, 2 dp) → list of MS3 entries
    # The MS2 precursor for an MS3 entry is msn_precursor_mzs[0];
    # the MS3 precursor (fragment re-isolated from MS2) is msn_precursor_mzs[1].
    ms3_by_ms2_prec: dict = defaultdict(list)
    for e in ms3_entries:
        msn_mzs = e.get("msn_precursor_mzs", [])
        if len(msn_mzs) < 2:
            continue
        ms2_prec = round(float(msn_mzs[0]), 2)
        ms3_prec = float(msn_mzs[1])
        ms3_by_ms2_prec[ms2_prec].append({
            "ms3_prec_mz":  ms3_prec,
            "ms1_prec_mz":  e.get("precursor_mz"),
            "num_peaks":    e.get("num_peaks", len(e.get("peaks", []))),
            "smiles":       e.get("smiles", ""),
            "feature_id":   e.get("feature_id", ""),
            "compound":     e.get("compound_name", ""),
        })

    print(f"\n  Unique MS2 precursors with ≥1 MS3 child : {len(ms3_by_ms2_prec):,}")
    print(f"\n--- Sample MSn Hierarchy (First {max_show} MS2 → MS3 linkages) ---\n")

    for ms2_prec_mz, children in list(ms3_by_ms2_prec.items())[:max_show]:
        first = children[0]
        print(f"MS1 Precursor m/z : {first['ms1_prec_mz']:.4f}")
        print(f"MS2 Precursor m/z : {ms2_prec_mz:.4f}  ({len(children)} MS3 child scan(s))")
        for ch in children:
            print(f"  └── MS3 Precursor m/z : {ch['ms3_prec_mz']:.4f}  "
                  f"| Peaks : {ch['num_peaks']:>3}  "
                  f"| SMILES : {ch['smiles'][:60]}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_msn_tree.py <path_to_*_msn.json>")
        sys.exit(1)
    extract_msn_tree(sys.argv[1])
