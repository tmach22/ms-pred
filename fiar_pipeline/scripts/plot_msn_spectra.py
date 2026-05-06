"""
MSn Hierarchical Spectra Plotter
==================================
Randomly samples N molecules from an MSnLib NDJSON pair (ms2 + msn) and
generates one stacked matplotlib figure per molecule showing:
  - Panel 0 : merged MS2 spectrum (blue)
  - Panel 1+ : one panel per unique MS3 precursor (orange shades)

Two-pass design keeps memory O(sampled molecules) regardless of file size.

Usage
-----
/data/nas-gpu/wang/tmach007/ms-pred/y/envs/ms-gen/bin/python \\
    fiar_pipeline/scripts/plot_msn_spectra.py \\
    --ms2_json /path/to/library_ms2.json \\
    --msn_json /path/to/library_msn.json \\
    --n_samples 5 \\
    --out_dir   /path/to/output/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────
MZ_ROUND     = 4          # decimal places for precursor_mz key equality
MS2_COLOR    = "#2979C8"  # steel blue
MS3_COLORS   = [          # cycling palette for successive MS3 panels
    "#E8660A", "#D44000", "#BF3000", "#A02000",
]
STEM_LINEWIDTH = 0.8
LABEL_FONTSIZE = 7


# ── NDJSON utilities ──────────────────────────────────────────────────────────

def _iter_ndjson(path: str):
    """Yield parsed dicts from a newline-delimited JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"  [WARN] {Path(path).name}:{lineno} JSON error: {exc}",
                      file=sys.stderr)


def _round_mz(v: Any) -> float | None:
    """Return precursor_mz rounded to MZ_ROUND dp, or None on failure."""
    try:
        return round(float(v), MZ_ROUND)
    except (TypeError, ValueError):
        return None


# ── Pass 1: discover all precursor_mz values that have MS3 chains ─────────────

def discover_valid_masses(msn_json: str) -> set[float]:
    """
    Stream msn_json and collect every precursor_mz (rounded to MZ_ROUND dp)
    for entries that satisfy:
      - ms_level == 3
      - len(msn_precursor_mzs) >= 2   (MS2 prec + MS3 prec both present)
    """
    valid: set[float] = set()
    for entry in _iter_ndjson(msn_json):
        if entry.get("ms_level") != 3:
            continue
        if len(entry.get("msn_precursor_mzs") or []) < 2:
            continue
        mz = _round_mz(entry.get("precursor_mz"))
        if mz is not None:
            valid.add(mz)
    return valid


# ── Pass 2: extract spectra for sampled masses ────────────────────────────────

def _normalise_peaks(peaks: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (mz_array, relative_intensity_0_to_100) from raw [[mz, int], ...]."""
    if not peaks:
        return np.array([]), np.array([])
    arr = np.array([[float(p[0]), float(p[1])] for p in peaks], dtype=np.float64)
    mzs  = arr[:, 0]
    ints = arr[:, 1]
    base = ints.max()
    if base > 0:
        ints = ints / base * 100.0
    return mzs, ints


def _median_ms3_ce(entry: dict) -> float | None:
    """Return median MS3 CE from msn_collision_energies or collision_energy."""
    pairs = entry.get("msn_collision_energies") or []
    ms3_ces = [float(p[1]) for p in pairs
               if isinstance(p, (list, tuple)) and len(p) >= 2
               and math.isfinite(float(p[1])) and float(p[1]) > 0]
    if ms3_ces:
        return float(np.median(ms3_ces))
    flat = entry.get("collision_energy") or []
    vals = [float(v) for v in (flat if isinstance(flat, list) else [flat])
            if math.isfinite(float(v)) and float(v) > 0]
    return float(np.median(vals)) if vals else None


def extract_spectra(
    ms2_json: str,
    msn_json: str,
    target_masses: set[float],
) -> dict[float, dict]:
    """
    Stream both files and collect:
      data[mz] = {
        "ms2"  : [entry, ...],
        "ms3"  : {ms3_prec_mz_rounded: [entry, ...], ...},
        "meta" : {"compound_name": ..., "adduct": ..., "smiles": ...},
      }
    """
    data: dict[float, dict] = {
        mz: {"ms2": [], "ms3": defaultdict(list), "meta": {}}
        for mz in target_masses
    }

    for entry in _iter_ndjson(ms2_json):
        if entry.get("ms_level") != 2:
            continue
        mz = _round_mz(entry.get("precursor_mz"))
        if mz in data:
            data[mz]["ms2"].append(entry)
            if not data[mz]["meta"]:
                data[mz]["meta"] = {
                    "compound_name": entry.get("compound_name", "Unknown"),
                    "adduct":        entry.get("adduct", ""),
                    "smiles":        entry.get("smiles", ""),
                    "polarity":      entry.get("polarity", ""),
                }

    for entry in _iter_ndjson(msn_json):
        if entry.get("ms_level") != 3:
            continue
        if len(entry.get("msn_precursor_mzs") or []) < 2:
            continue
        mz = _round_mz(entry.get("precursor_mz"))
        if mz not in data:
            continue
        ms3_prec = _round_mz(entry["msn_precursor_mzs"][1])
        if ms3_prec is not None:
            data[mz]["ms3"][ms3_prec].append(entry)
            # Backfill meta from MSn if MS2 file didn't provide it
            if not data[mz]["meta"]:
                data[mz]["meta"] = {
                    "compound_name": entry.get("compound_name", "Unknown"),
                    "adduct":        entry.get("adduct", ""),
                    "smiles":        entry.get("smiles", ""),
                    "polarity":      entry.get("polarity", ""),
                }

    return data


# ── Plotting ──────────────────────────────────────────────────────────────────

def _stem_panel(
    ax: plt.Axes,
    mzs: np.ndarray,
    ints: np.ndarray,
    color: str,
    title: str,
    precursor_mz: float | None = None,
) -> None:
    """Draw a single mass spectrum panel using vertical stems."""
    ax.set_facecolor("#F9F9F9")

    if len(mzs) == 0:
        ax.text(0.5, 0.5, "No peaks", ha="center", va="center",
                transform=ax.transAxes, color="grey", fontsize=9)
    else:
        markerline, stemlines, baseline = ax.stem(
            mzs, ints, linefmt=color, markerfmt=" ", basefmt="black"
        )
        stemlines.set_linewidth(STEM_LINEWIDTH)
        baseline.set_linewidth(0.6)

        # Annotate top-5 peaks by intensity
        top5_idx = np.argsort(ints)[-5:][::-1]
        for i in top5_idx:
            ax.annotate(
                f"{mzs[i]:.4f}",
                xy=(mzs[i], ints[i]),
                xytext=(2, 3),
                textcoords="offset points",
                fontsize=LABEL_FONTSIZE,
                color="#333333",
                ha="left",
            )

    if precursor_mz is not None:
        ax.axvline(precursor_mz, color="red", linewidth=0.8,
                   linestyle="--", alpha=0.6, label=f"prec {precursor_mz:.4f}")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.6)

    ax.set_title(title, fontsize=8, pad=4)
    ax.set_ylabel("Rel. Intensity (%)", fontsize=7)
    ax.set_ylim(-5, 115)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_molecule(
    prec_mz: float,
    mol_data: dict,
    out_dir: Path,
) -> None:
    """
    Generate and save the stacked MS2 / MS3 figure for one molecule.
    Closes the figure after saving to prevent memory leaks.
    """
    meta = mol_data["meta"]
    ms2_entries = mol_data["ms2"]
    ms3_groups  = mol_data["ms3"]   # {ms3_prec_mz: [entries]}

    compound = meta.get("compound_name", "Unknown")
    adduct   = meta.get("adduct", "")
    smiles   = meta.get("smiles", "")

    # Build panels: MS2 first, then one per unique MS3 precursor (sorted)
    ms3_precs = sorted(ms3_groups.keys())
    n_panels  = 1 + len(ms3_precs)

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(12, 3.5 * n_panels),
        sharex=False,
    )
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f"{compound}   {adduct}   prec m/z {prec_mz:.4f}\n"
        f"{smiles[:90]}{'…' if len(smiles) > 90 else ''}",
        fontsize=9, y=1.01,
    )

    # ── MS2 panel ─────────────────────────────────────────────────────────────
    if ms2_entries:
        # Merge all MS2 peaks (union of peaks across CE-merged entries); use
        # the entry with the highest peak count as the representative spectrum.
        rep_ms2 = max(ms2_entries, key=lambda e: len(e.get("peaks") or []))
        mzs, ints = _normalise_peaks(rep_ms2.get("peaks") or [])
        ce_list = rep_ms2.get("collision_energy") or []
        ce_str  = "/".join(f"{v:.0f}" for v in
                           (ce_list if isinstance(ce_list, list) else [ce_list]))
        ms2_title = f"MS2  |  prec {prec_mz:.4f}  |  CE {ce_str} eV"
    else:
        mzs, ints = np.array([]), np.array([])
        ms2_title = f"MS2  |  prec {prec_mz:.4f}  |  (no data)"

    _stem_panel(axes[0], mzs, ints, MS2_COLOR, ms2_title,
                precursor_mz=prec_mz)

    # ── MS3 panels ────────────────────────────────────────────────────────────
    for panel_idx, ms3_prec in enumerate(ms3_precs, start=1):
        entries = ms3_groups[ms3_prec]
        color   = MS3_COLORS[panel_idx % len(MS3_COLORS)]

        # Representative entry: most peaks
        rep = max(entries, key=lambda e: len(e.get("peaks") or []))
        mzs, ints = _normalise_peaks(rep.get("peaks") or [])
        ce = _median_ms3_ce(rep)
        ce_str = f"{ce:.0f}" if ce is not None else "?"

        title = (
            f"MS3  |  MS2 prec {prec_mz:.4f}  →  MS3 prec {ms3_prec:.4f}"
            f"  |  median CE {ce_str} eV"
            f"  ({len(entries)} scan{'s' if len(entries)>1 else ''})"
        )
        _stem_panel(axes[panel_idx], mzs, ints, color, title,
                    precursor_mz=ms3_prec)

    for ax in axes:
        ax.set_xlabel("m/z", fontsize=7)

    fig.tight_layout()

    fname = out_dir / f"spectra_{prec_mz:.4f}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot randomly sampled MSn hierarchical spectra from MSnLib NDJSON files."
    )
    p.add_argument("--ms2_json",  required=True,
                   help="Path to the *_ms2.json NDJSON file")
    p.add_argument("--msn_json",  required=True,
                   help="Path to the *_msn.json NDJSON file")
    p.add_argument("--n_samples", type=int, default=3,
                   help="Number of molecules to sample (default: 3)")
    p.add_argument("--seed",      type=int, default=None,
                   help="Random seed for reproducible sampling")
    p.add_argument("--out_dir",   default="sample_plots",
                   help="Output directory for PNG files (default: sample_plots)")
    return p.parse_args()


def main() -> None:
    args = _cli()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)

    # ── Pass 1: discover valid masses ─────────────────────────────────────────
    print(f"[Pass 1] Scanning for valid MS³ chains: {Path(args.msn_json).name}")
    valid_masses = discover_valid_masses(args.msn_json)
    print(f"         {len(valid_masses):,} unique precursor m/z values with MS³ chains")

    if not valid_masses:
        print("[!] No valid MS³ entries found. Check the --msn_json file.")
        sys.exit(1)

    n = min(args.n_samples, len(valid_masses))
    if n < args.n_samples:
        print(f"[!] Requested {args.n_samples} samples but only {len(valid_masses)} "
              f"available — sampling all {n}.")
    sampled = set(random.sample(sorted(valid_masses), n))
    print(f"\n[Sample]  Selected {n} masses:")
    for mz in sorted(sampled):
        print(f"          {mz:.4f}")

    # ── Pass 2: extract spectra for sampled masses ────────────────────────────
    print(f"\n[Pass 2] Extracting spectra from both files …")
    mol_data = extract_spectra(args.ms2_json, args.msn_json, sampled)

    # ── Plotting ──────────────────────────────────────────────────────────────
    print(f"\n[Plot]   Generating figures → {out_dir}/")
    for mz in sorted(sampled):
        compound = mol_data[mz]["meta"].get("compound_name", "Unknown")
        n_ms2    = len(mol_data[mz]["ms2"])
        n_ms3    = sum(len(v) for v in mol_data[mz]["ms3"].values())
        print(f"\n  m/z {mz:.4f}  |  {compound}")
        print(f"    MS2 entries: {n_ms2}  |  MS3 entries: {n_ms3}  "
              f"({len(mol_data[mz]['ms3'])} unique MS3 precursors)")
        plot_molecule(mz, mol_data[mz], out_dir)

    print(f"\n[Done]   {n} figure(s) saved to {out_dir}/")


if __name__ == "__main__":
    main()
