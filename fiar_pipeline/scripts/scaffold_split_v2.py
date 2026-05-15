import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import logging
import multiprocessing as mp
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_murcko_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return "INVALID"
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold if scaffold else "LINEAR"
    except Exception:
        return "ERROR"

def perform_scaffold_split(df, train_frac=0.8, val_frac=0.1, test_frac=0.1, workers=24):
    assert np.isclose(train_frac + val_frac + test_frac, 1.0)

    unique_smiles = df['intact_parent_smiles'].unique()
    logging.info(f"Extracting Murcko Scaffolds for {len(unique_smiles)} unique molecules using {workers} CPUs...")

    with mp.Pool(processes=workers) as pool:
        scaffolds = list(tqdm(pool.imap(get_murcko_scaffold, unique_smiles),
                              total=len(unique_smiles), desc="Calculating Scaffolds"))

    scaffold_dict = dict(zip(unique_smiles, scaffolds))
    df['scaffold'] = df['intact_parent_smiles'].map(scaffold_dict)

    initial_len = len(df)
    df = df[~df['scaffold'].isin(["INVALID", "ERROR"])]
    if len(df) < initial_len:
        logging.warning(f"Dropped {initial_len - len(df)} rows due to RDKit parsing errors.")

    logging.info("Assigning scaffolds to partitions (Fat Head -> Long Tail)...")
    scaffold_sizes = df.groupby('scaffold').size().sort_values(ascending=False)

    train_cutoff = int(train_frac * len(df))
    val_cutoff = int((train_frac + val_frac) * len(df))

    train_scaffolds, val_scaffolds, test_scaffolds = set(), set(), set()
    current_count = 0

    for scaffold, size in tqdm(scaffold_sizes.items(), desc="Partitioning"):
        if current_count < train_cutoff: train_scaffolds.add(scaffold)
        elif current_count < val_cutoff: val_scaffolds.add(scaffold)
        else: test_scaffolds.add(scaffold)
        current_count += size

    logging.info("Slicing dataframes...")
    train_df = df[df['scaffold'].isin(train_scaffolds)].drop(columns=['scaffold'])
    val_df = df[df['scaffold'].isin(val_scaffolds)].drop(columns=['scaffold'])
    test_df = df[df['scaffold'].isin(test_scaffolds)].drop(columns=['scaffold'])

    return train_df, val_df, test_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    in_path, out_path = Path(args.input_dir), Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = list(in_path.glob("*.parquet"))
    logging.info(f"Loading {len(files)} parquet files into RAM...")

    dfs = [pd.read_parquet(f) for f in tqdm(files, desc="Reading Parquets")]
    df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Loaded {len(df)} MS3 tensors.")

    train_df, val_df, test_df = perform_scaffold_split(df, workers=args.workers)

    logging.info(f"Saving splits to {out_path}...")
    train_df.to_parquet(out_path / "train.parquet", index=False, engine="pyarrow")
    val_df.to_parquet(out_path / "val.parquet", index=False, engine="pyarrow")
    test_df.to_parquet(out_path / "test.parquet", index=False, engine="pyarrow")

    print("\n=== FINAL SPLIT SUMMARY ===")
    print(f"Train : {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val   : {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test  : {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)\n")

if __name__ == "__main__":
    main()
