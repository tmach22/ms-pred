import pandas as pd
import argparse
import sys
import os

def prepare_tensors(clean_spectra_path, master_index_path, out_dir):
    # 1. Load the Stage 1 clean MS3 arrays
    spectra_df = pd.read_parquet(clean_spectra_path)

    # 2. Load the Master Metadata Index
    index_df = pd.read_parquet(master_index_path)

    # Extract the base .mzML filename to match the index
    base_file = os.path.basename(clean_spectra_path).replace('clean_spectra_', '').replace('.parquet', '.mzML')
    file_index = index_df[index_df['mzml_file'] == base_file]

    if file_index.empty:
        print(f"Warning: No metadata found for {base_file}.")
        return

    # 3. Merge on scan_number
    merged_df = pd.merge(spectra_df, file_index, on='scan_number', how='inner')

    if merged_df.empty:
        return

    # 4. Select and rename ONLY the columns needed for the Distillation Training Loop
    final_df = merged_df[[
        'smiles',              # The intact MS1 molecule (Input to Actor)
        'ms2_precursor_mz',    # The isolated MS2 mass (The Filter)
        'collision_energy',    # The thermodynamic context
        'ms3_mz',              # The empirical MS3 masses (Target for Oracle)
        'ms3_intensity'        # The empirical MS3 intensities (Target for Oracle)
    ]].rename(columns={'smiles': 'intact_parent_smiles'})

    # 5. Save the final Distillation Tensors
    out_name = os.path.basename(clean_spectra_path).replace('clean_spectra_', 'distillation_tensors_')
    out_path = os.path.join(out_dir, out_name)
    final_df.to_parquet(out_path, index=False, engine='pyarrow')
    print(f"Merged {len(final_df)} tensors -> {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    prepare_tensors(args.spectra, args.index, args.outdir)
