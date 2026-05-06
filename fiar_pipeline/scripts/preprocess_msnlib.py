"""
MSnLib MS³ Edge Preprocessor
==============================
Parses all *_msn.json (NDJSON) files from the MSnLib Zenodo deposit,
extracts MS³ precursor chains, applies quality filters, and writes a
single ms3_edge_labels.feather for downstream edge-supervision training.

Field mapping (verified against actual Zenodo 11163380 schema)
--------------------------------------------------------------
  ms_level               : int   — must be 3
  precursor_mz           : float — MS1 precursor
  msn_precursor_mzs      : list  — [MS2_prec_mz, MS3_prec_mz, ...]
  msn_collision_energies : list  — [[ms2_ce, ms3_ce], ...]   (per-scan pairs)
  collision_energy        : list  — fallback flat CE list
  precursor_purity       : float — isolation purity  (gate ≥ 0.70)
  smiles                 : str
  polarity               : str   — '+' | '-'

Output schema
-------------
  smiles      str
  parent_mz   float  (MS2 precursor m/z)
  child_mz    float  (MS3 precursor m/z)
  nce         float  (median MS3 CE / 50.0)
  purity      float
  ion_mode    str    ('positive' | 'negative')

Usage
-----
cd /data/nas-gpu/wang/tmach007/ms-pred

/data/nas-gpu/wang/tmach007/ms-pred/y/envs/ms-gen/bin/python \\
    fiar_pipeline/scripts/preprocess_msnlib.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
JSON_DIR = Path("/data/nas-gpu/wang/tmach007/ms-pred/data/MSnLib/libraries/json")
OUTPUT   = Path("/data/nas-gpu/wang/tmach007/ms-pred/data/ms3_edge_labels.feather")

# ── Thresholds ────────────────────────────────────────────────────────────────
PURITY_GATE  = 0.70
CE_REFERENCE = 50.0          # NCE normalisation divisor


# ── Helpers ───────────────────────────────────────────────────────────────────

_POLARITY_MAP = {
    "+": "positive", "pos": "positive", "positive": "positive",
    "-": "negative", "neg": "negative", "negative": "negative",
}


def _normalise_polarity(raw: Any) -> str:
    return _POLARITY_MAP.get(str(raw).strip().lower(), "unknown")


def _median_ms3_ce(entry: dict) -> float | None:
    """
    Extract the representative MS3 collision energy.

    Primary  — msn_collision_energies: [[ms2_ce, ms3_ce], ...]
               Take the ms3_ce (index 1) from every pair, then median.
    Fallback — collision_energy: [ce1, ce2, ...]   flat list.
               Assumed to contain only MS3 CEs for ms_level==3 entries.
    Returns None if no valid CE is found.
    """
    pairs = entry.get("msn_collision_energies")
    if pairs:
        ms3_ces = []
        for pair in pairs:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                try:
                    v = float(pair[1])
                    if math.isfinite(v) and v > 0:
                        ms3_ces.append(v)
                except (TypeError, ValueError):
                    pass
        if ms3_ces:
            return statistics.median(ms3_ces)

    flat = entry.get("collision_energy")
    if flat:
        vals = []
        for v in (flat if isinstance(flat, list) else [flat]):
            try:
                fv = float(v)
                if math.isfinite(fv) and fv > 0:
                    vals.append(fv)
            except (TypeError, ValueError):
                pass
        if vals:
            return statistics.median(vals)

    return None


def _parse_entry(entry: dict) -> dict | None:
    """
    Return a record dict for a valid MS³ entry, or None if filtered out.

    Filters applied
    ---------------
    1. ms_level == 3
    2. Full precursor chain length ≥ 3
       chain = [precursor_mz, *msn_precursor_mzs]
    3. precursor_purity ≥ PURITY_GATE
    4. smiles must be a non-empty string
    5. CE must be parseable and > 0
    """
    # ── MS level ──────────────────────────────────────────────────────────────
    if entry.get("ms_level") != 3:
        return None

    # ── Precursor chain ───────────────────────────────────────────────────────
    prec_mz = entry.get("precursor_mz")
    msn_mzs = entry.get("msn_precursor_mzs", [])
    if not isinstance(msn_mzs, list):
        return None

    try:
        chain = [float(prec_mz)] + [float(m) for m in msn_mzs]
    except (TypeError, ValueError):
        return None

    if len(chain) < 3:
        return None

    parent_mz = chain[1]   # MS2 precursor
    child_mz  = chain[2]   # MS3 precursor

    # ── Purity gate ───────────────────────────────────────────────────────────
    try:
        purity = float(entry.get("precursor_purity", 0.0))
    except (TypeError, ValueError):
        return None
    if purity < PURITY_GATE:
        return None

    # ── SMILES ────────────────────────────────────────────────────────────────
    smiles = entry.get("smiles") or entry.get("SMILES") or ""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    smiles = smiles.strip()

    # ── Collision energy → NCE ────────────────────────────────────────────────
    ce = _median_ms3_ce(entry)
    if ce is None:
        return None
    nce = ce / CE_REFERENCE

    # ── Ion mode ──────────────────────────────────────────────────────────────
    ion_mode = _normalise_polarity(entry.get("polarity", ""))

    return dict(
        smiles    = smiles,
        parent_mz = parent_mz,
        child_mz  = child_mz,
        nce       = nce,
        purity    = purity,
        ion_mode  = ion_mode,
    )


# ── File-level parser ─────────────────────────────────────────────────────────

def parse_ndjson_file(path: Path) -> list[dict]:
    """
    Read one NDJSON file and return all valid MS³ records.
    Skips blank lines and malformed JSON lines with a warning.
    """
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"  [WARN] {path.name}:{lineno} — JSON parse error: {exc}",
                      file=sys.stderr)
                continue

            rec = _parse_entry(entry)
            if rec is not None:
                records.append(rec)
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Discover files ────────────────────────────────────────────────────────
    msn_files = sorted(JSON_DIR.glob("*_msn.json"))
    if not msn_files:
        print(f"[!] No *_msn.json files found in:\n    {JSON_DIR}")
        sys.exit(1)

    print(f"[Preprocess] Found {len(msn_files)} *_msn.json file(s) in:")
    print(f"             {JSON_DIR}\n")

    # ── Parse ─────────────────────────────────────────────────────────────────
    all_records: list[dict] = []
    file_stats: list[tuple[str, int]] = []

    for path in tqdm(msn_files, desc="Files", unit="file"):
        recs = parse_ndjson_file(path)
        file_stats.append((path.name, len(recs)))
        all_records.extend(recs)

    # ── Per-file summary ──────────────────────────────────────────────────────
    print("\n[Preprocess] Per-file MS³ edge counts:")
    print(f"  {'Filename':<50}  {'Edges':>7}")
    print(f"  {'-'*50}  {'-'*7}")
    for fname, n in file_stats:
        print(f"  {fname:<50}  {n:>7,}")
    print(f"  {'TOTAL':<50}  {len(all_records):>7,}")

    if not all_records:
        print("\n[!] No valid MS³ edges extracted — check purity threshold "
              "and file contents.")
        sys.exit(1)

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame(all_records, columns=[
        "smiles", "parent_mz", "child_mz", "nce", "purity", "ion_mode"
    ])

    # Cast to compact dtypes
    df["parent_mz"] = df["parent_mz"].astype("float32")
    df["child_mz"]  = df["child_mz"].astype("float32")
    df["nce"]       = df["nce"].astype("float32")
    df["purity"]    = df["purity"].astype("float32")
    df["ion_mode"]  = df["ion_mode"].astype("category")

    # ── Quick stats ───────────────────────────────────────────────────────────
    print(f"\n[Preprocess] Dataset stats:")
    print(f"  Total MS³ edges : {len(df):,}")
    print(f"  Unique SMILES   : {df['smiles'].nunique():,}")
    print(f"  Positive ions   : {(df['ion_mode']=='positive').sum():,}")
    print(f"  Negative ions   : {(df['ion_mode']=='negative').sum():,}")
    print(f"  Mean purity     : {df['purity'].mean():.3f}")
    print(f"  NCE range       : {df['nce'].min():.3f} – {df['nce'].max():.3f}")

    # ── Export ────────────────────────────────────────────────────────────────
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_feather(OUTPUT)
    print(f"\n[Preprocess] Saved {len(df):,} edges → {OUTPUT}")


if __name__ == "__main__":
    main()
