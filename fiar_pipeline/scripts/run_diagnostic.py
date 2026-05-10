import pyopenms as ms
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR = _REPO_ROOT / "fiar_pipeline" / "results" / "diagnostics"

def run_diagnostic(mzml_file_path):
    print(f"Loading {mzml_file_path}...")
    exp = ms.MSExperiment()
    ms.MzMLFile().load(mzml_file_path, exp)

    target_scan = None
    ms2_precursor_mz = 0.0

    # 1. Hunt for a rich MS3 scan
    for spec in exp:
        if spec.getMSLevel() == 3:
            mz_array, int_array = spec.get_peaks()
            # Find a scan with enough peaks to see the effect of filtering
            if len(mz_array) > 30:
                target_scan = spec
                # The MS2 precursor is the parent isolated to make this MS3 scan
                ms2_precursor_mz = spec.getPrecursors()[0].getMZ()
                break

    if target_scan is None:
        print("No suitable MS3 scan found in this file.")
        sys.exit(0)

    raw_mz, raw_int = target_scan.get_peaks()

    # --- The ICEBERG Transformations ---

    # Step A: Physical Impossibility Pruning (Cannot be heavier than parent)
    # Adding a 1.5 Da tolerance to account for isotopic envelopes (e.g., 13C)
    valid_indices = raw_mz <= (ms2_precursor_mz + 1.5)
    mz_pruned = raw_mz[valid_indices]
    int_pruned = raw_int[valid_indices]

    # Step B: Normalization & Square Root Transformation
    if len(int_pruned) == 0:
        print("All peaks pruned. MS3 scan was invalid.")
        sys.exit(0)

    base_peak_int = np.max(int_pruned)
    int_norm = int_pruned / base_peak_int
    int_sqrt = np.sqrt(int_norm)

    # Step C: Top-50 Thresholding
    if len(mz_pruned) > 50:
        top_50_indices = np.argsort(int_sqrt)[-50:]
        final_mz = mz_pruned[top_50_indices]
        final_int = int_sqrt[top_50_indices]
    else:
        final_mz = mz_pruned
        final_int = int_sqrt

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Raw Absolute Intensity
    axes[0].vlines(raw_mz, 0, raw_int, color='gray', alpha=0.7)
    axes[0].set_title(f"Raw MS3 Spectrum (MS2 Parent: {ms2_precursor_mz:.2f} Da)")
    axes[0].set_ylabel("Absolute Intensity (Orbitrap)")
    axes[0].set_xlabel("m/z")

    # Plot 2: ICEBERG Transformed
    axes[1].vlines(final_mz, 0, final_int, color='blue', linewidth=1.5)
    axes[1].set_title("ICEBERG Transformed (Sqrt Norm, Top 50, Pruned)")
    axes[1].set_ylabel("Intensity ($\\sqrt{I/I_{max}}$)")
    axes[1].set_xlabel("m/z")
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_img = _OUTPUT_DIR / "transformation_diagnostic.png"
    plt.savefig(output_img, dpi=300)
    print(f"Diagnostic plot saved successfully to {output_img}")
    print(f"Transformation metrics:")
    print(f"  - Original raw peaks: {len(raw_mz)}")
    print(f"  - Peaks after pruning impossible masses: {len(mz_pruned)}")
    print(f"  - Final tensor size after Top-50 cutoff: {len(final_mz)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_diagnostic.py <path_to_mzml>")
        sys.exit(1)

    target_file = sys.argv[1]
    run_diagnostic(target_file)
