"""
Reward-Scaled DataLoader
========================
Reads the output of offline_oracle_scorer.py and batches it for the
active Reward-Scaled Teacher Forcing training loop.

Usage (standalone test)
-----------------------
cd /data/nas-gpu/wang/tmach007/ms-pred
python reward_scaled_dataloader.py
"""

import ast
import glob
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class RewardScaledDataset(Dataset):
    def __init__(self, parquet_dir: str):
        print(f"[DataLoader] Loading Reward-Scaled Tensors from {parquet_dir}...")
        files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

        dfs = [pd.read_parquet(f) for f in sorted(files)]
        self.data_df = pd.concat(dfs, ignore_index=True)
        print(f"[DataLoader] Loaded {len(self.data_df)} training examples.")

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int):
        row = self.data_df.iloc[idx]

        # 1. Actor input: MS1 SMILES (featurized in the training loop)
        smiles = row["ms1_intact_smiles"]

        # 2. Target output: ground-truth MS2 formula string
        target_formula = row["target_ms2_formula"]

        # 3. Offline Reward Scalar R ∈ (0, 1]
        reward = torch.tensor([float(row["offline_reward_scalar"])], dtype=torch.float32)

        # 4. Thermodynamic context: collision energy
        ce_val = row["collision_energy"]
        if isinstance(ce_val, str):
            try:
                ce_list = ast.literal_eval(ce_val)
                ce_val = sum(ce_list) / len(ce_list) if ce_list else 0.0
            except Exception:
                try:
                    ce_val = float(ce_val)
                except Exception:
                    ce_val = 0.0
        elif isinstance(ce_val, (list, tuple)):
            ce_val = sum(ce_val) / len(ce_val) if ce_val else 0.0
        else:
            ce_val = float(ce_val) if ce_val is not None else 0.0

        ce = torch.tensor([ce_val], dtype=torch.float32)

        return smiles, target_formula, reward, ce


def reward_scaled_collate(batch):
    """Collates Actor-Oracle inputs into batched tensors."""
    smiles_list  = [item[0] for item in batch]
    formula_list = [item[1] for item in batch]
    rewards      = torch.cat([item[2] for item in batch], dim=0)
    ces          = torch.cat([item[3] for item in batch], dim=0)
    return smiles_list, formula_list, rewards, ces


if __name__ == "__main__":
    print("\n--- Initiating Reward-Scaled DataLoader Test ---")
    TARGET_DIR = "/data/nas-gpu/wang/tmach007/ms-pred/data/MSnLib/reward_scaled_tensors"

    try:
        dataset = RewardScaledDataset(parquet_dir=TARGET_DIR)
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=reward_scaled_collate,
        )

        graphs, formulas, rewards, ces = next(iter(dataloader))

        print("\n[SUCCESS] Batch successfully constructed!")
        print(f"  Batch Size              : {len(graphs)}")
        print(f"  Actor Input (SMILES)    : {graphs}")
        print(f"  Target Formulas         : {formulas}")
        print(f"  Reward Scalars shape    : {rewards.shape}  → {rewards.squeeze().tolist()}")
        print(f"  Collision Energy shape  : {ces.shape}      → {ces.squeeze().tolist()}")

        # Validate reward bounds
        assert rewards.min() >= 0.0 and rewards.max() <= 1.0, \
            f"Reward out of [0,1] bounds: min={rewards.min():.4f} max={rewards.max():.4f}"
        print(f"\n[Validation] Reward scalars strictly bounded in [0.0, 1.0] ✓")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("  → Run offline_oracle_scorer.py first to generate reward_scaled_tensors.")
    except Exception as e:
        print(f"\n[ERROR] DataLoader test failed: {e}")
        raise
