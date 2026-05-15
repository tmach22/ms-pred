import ast
import time
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class MS3DistillationDataset(Dataset):
    def __init__(self, parquet_path: str):
        print(f"Loading dataset from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        print(f"Dataset loaded with {len(self.df):,} rows.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        smiles = row['intact_parent_smiles']

        # collision_energy is stored as a stringified list e.g. "[20.0, 30.0, 60.0]"
        raw_ce = row.get('collision_energy', '30.0')
        try:
            ce_val = float(ast.literal_eval(raw_ce)[0]) if isinstance(raw_ce, str) and raw_ce.startswith('[') else float(raw_ce)
        except Exception:
            ce_val = 30.0
        ce = torch.tensor(ce_val, dtype=torch.float32)

        ms2_mz = torch.tensor(float(row['ms2_precursor_mz']), dtype=torch.float32)
        ms3_mz = torch.tensor(row['ms3_mz'], dtype=torch.float32)
        ms3_int = torch.tensor(row['ms3_intensity'], dtype=torch.float32)

        return smiles, ce, ms2_mz, ms3_mz, ms3_int


def collate_actor_oracle(batch):
    """Collates SMILES strings and pads variable-length MS3 arrays."""
    smiles_list, ces, ms2_mzs, ms3_mzs, ms3_ints = map(list, zip(*batch))

    batched_ces    = torch.stack(ces)
    batched_ms2_mz = torch.stack(ms2_mzs)

    padded_ms3_mz  = torch.nn.utils.rnn.pad_sequence(ms3_mzs,  batch_first=True, padding_value=0.0)
    padded_ms3_int = torch.nn.utils.rnn.pad_sequence(ms3_ints, batch_first=True, padding_value=0.0)

    return smiles_list, batched_ces, batched_ms2_mz, padded_ms3_mz, padded_ms3_int


if __name__ == "__main__":
    TEST_PARQUET = "/data/nas-gpu/wang/tmach007/ms-pred/data/MSnLib/splits_v2/test.parquet"

    dataset = MS3DistillationDataset(TEST_PARQUET)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_actor_oracle,
        num_workers=4,
        pin_memory=True,
    )

    print("\nStarting DataLoader fetch test...")
    start_time = time.time()

    for batch_idx, batch_data in enumerate(dataloader):
        smiles_list, batched_ces, batched_ms2_mz, padded_ms3_mz, padded_ms3_int = batch_data

        print(f"\n=== BATCH {batch_idx} FETCH SUCCESSFUL ===")
        print(f"Time to fetch: {time.time() - start_time:.2f} seconds")
        print(f"SMILES list length : {len(smiles_list)}")
        print(f"Sample SMILES      : {smiles_list[0][:60]}")
        print(f"CEs                : {batched_ces.shape}  -> {batched_ces[:3]}")
        print(f"MS2 m/z            : {batched_ms2_mz.shape} -> {batched_ms2_mz[:3]}")
        print(f"Padded MS3 m/z     : {padded_ms3_mz.shape}")
        print(f"Padded MS3 int     : {padded_ms3_int.shape}")
        print("=====================================")
        break
