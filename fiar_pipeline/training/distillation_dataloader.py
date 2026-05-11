import ast
import glob
import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class ActorOracleDataset(Dataset):
    def __init__(self, parquet_dir):
        """
        Loads the entire distillation dataset into memory for zero-latency batching.
        """
        print(f"Loading Distillation Tensors from {parquet_dir}...")
        files = glob.glob(os.path.join(parquet_dir, "*.parquet"))

        if not files:
            raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

        dfs = [pd.read_parquet(f) for f in files]
        self.data_df = pd.concat(dfs, ignore_index=True)
        print(f"Successfully loaded {len(self.data_df)} training examples into memory.")

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        # 1. The Actor Input (MS1 Graph)
        # Note: We return the SMILES string for now. The ICEBERG graph featurizer
        # (e.g., smiles_to_graph) will be applied here once integrated into the main loop.
        smiles = row['intact_parent_smiles']

        # 2. Thermodynamic Context (The Filter)
        # collision_energy is stored as a stringified list "[20.0, 45.0, 60.0]";
        # reduce to a scalar mean CE so context tensors are rectangular across samples.
        ce_raw = row['collision_energy']
        if isinstance(ce_raw, str):
            ce_list = ast.literal_eval(ce_raw)
        elif hasattr(ce_raw, '__iter__'):
            ce_list = list(ce_raw)
        else:
            ce_list = [float(ce_raw)]
        ce_scalar = float(sum(ce_list) / len(ce_list))

        context = {
            'collision_energy': torch.tensor([ce_scalar], dtype=torch.float32),
            'ms2_precursor_mz': torch.tensor([row['ms2_precursor_mz']], dtype=torch.float32)
        }

        # 3. The Oracle Targets (MS3 Ground Truth)
        targets = {
            'ms3_mz':       torch.tensor(row['ms3_mz'],       dtype=torch.float32),
            'ms3_intensity': torch.tensor(row['ms3_intensity'], dtype=torch.float32)
        }

        return smiles, context, targets


def actor_oracle_collate(batch):
    """
    Custom collate function required to handle variable-length MS3 arrays.
    Standard PyTorch dataloaders crash when stacking tensors of different sizes.
    """
    graphs = []
    collision_energies = []
    ms2_precursor_mzs = []
    ms3_mzs = []
    ms3_intensities = []

    for graph, context, targets in batch:
        graphs.append(graph)
        collision_energies.append(context['collision_energy'])
        ms2_precursor_mzs.append(context['ms2_precursor_mz'])

        # Keep variable-length targets as lists of tensors
        ms3_mzs.append(targets['ms3_mz'])
        ms3_intensities.append(targets['ms3_intensity'])

    # Batch the Context as standard rectangular tensors
    batched_context = {
        'collision_energy': torch.cat(collision_energies, dim=0),
        'ms2_precursor_mz': torch.cat(ms2_precursor_mzs, dim=0)
    }

    # Batch the Targets as lists (The Oracle will evaluate these sequentially or via padding later)
    batched_targets = {
        'ms3_mz':       ms3_mzs,
        'ms3_intensity': ms3_intensities
    }

    # Pass the graphs (SMILES strings for now) as a list.
    # When integrating PyG or DGL, use Batch.from_data_list(graphs) here.
    batched_graphs = graphs

    return batched_graphs, batched_context, batched_targets


if __name__ == "__main__":
    import pathlib
    _REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
    TARGET_DIR = str(_REPO_ROOT / "data" / "MSnLib" / "distillation_tensors")

    print("\n--- Initiating Actor-Oracle DataLoader Test ---")

    try:
        # 1. Instantiate the Dataset
        dataset = ActorOracleDataset(parquet_dir=TARGET_DIR)

        # 2. Instantiate the DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=actor_oracle_collate,
            pin_memory=True   # Optimizes GPU memory transfer
        )

        # 3. Fetch one batch
        graphs, context, targets = next(iter(dataloader))

        # 4. Report Tensor Shapes
        print("\n[SUCCESS] Batch successfully constructed!")
        print(f"Batch Size: {len(graphs)}")
        print("\n--- Input & Context Tensors ---")
        print(f"Actor Input (Graphs): List of {len(graphs)} elements (currently SMILES)")
        print(f"Context -> collision_energy: {context['collision_energy'].shape}")
        print(f"Context -> ms2_precursor_mz: {context['ms2_precursor_mz'].shape}")

        print(f"\n--- Oracle Target Tensors (Variable Lengths) ---")
        for i, mz_tensor in enumerate(targets['ms3_mz']):
            intensity_tensor = targets['ms3_intensity'][i]
            print(f"Sample {i} | MS3 M/Z shape: {mz_tensor.shape} | MS3 Intensity shape: {intensity_tensor.shape}")

    except Exception as e:
        import traceback
        print(f"\n[FATAL ERROR] DataLoader test failed: {e}")
        traceback.print_exc()
