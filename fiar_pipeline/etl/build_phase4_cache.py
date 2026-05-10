"""
Phase 4 ETL — Precompute 3D Distance Matrices & Thermo States
Run this once to eliminate the ETKDGv3 CPU bottleneck during training.
"""
import pandas as pd
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm
from joblib import Parallel, delayed

RDLogger.DisableLog("rdApp.*")
MAX_ATOMS = 100

def _extract_features(smiles: str):
    dummy_dist = torch.zeros(MAX_ATOMS, MAX_ATOMS, dtype=torch.float32)
    thermo_state = torch.tensor([0.0, 0.0], dtype=torch.float32)
    
    if not smiles or not isinstance(smiles, str):
        return smiles, thermo_state, dummy_dist

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, thermo_state, dummy_dist

        # ── 1. Thermo State (Calculated on implicit H graph) ──
        fc = sum(a.GetFormalCharge() for a in mol.GetAtoms())
        formal_charge = float(np.clip(fc / 4.0, -1.0, 1.0))
        
        has_quat_n = 0.0
        for atom in mol.GetAtoms():
            if (atom.GetAtomicNum() == 7 and atom.GetFormalCharge() > 0 
                and atom.GetTotalDegree() == 4 and atom.GetTotalNumHs() == 0):
                has_quat_n = 1.0
                break
        thermo_state = torch.tensor([formal_charge, has_quat_n], dtype=torch.float32)

        # ── 2. 3D Distance Matrix (The Add-and-Strip Fix) ──
        # Add explicit Hs so ETKDGv3 can fold it with accurate steric physics
        mol_3d = Chem.AddHs(mol)
        
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        ret = AllChem.EmbedMolecule(mol_3d, params)
        
        if ret == -1:
            ret = AllChem.EmbedMolecule(mol_3d, AllChem.ETDG())
            
        if ret != -1:
            # Strip the Hs back out. RDKit beautifully preserves the heavy-atom 
            # 3D coordinates, meaning our N x N tensor will perfectly match the GNN!
            mol_heavy_only = Chem.RemoveHs(mol_3d)
            
            conf = mol_heavy_only.GetConformer()
            positions = np.array(conf.GetPositions(), dtype=np.float32)
            N = positions.shape[0]
            
            diff = positions[:, None, :] - positions[None, :, :]
            dist = np.sqrt((diff * diff).sum(-1))
            
            max_d = dist.max()
            if max_d > 0.0:
                dist /= max_d
                
            n_fill = min(N, MAX_ATOMS)
            dummy_dist[:n_fill, :n_fill] = torch.from_numpy(dist[:n_fill, :n_fill])

    except Exception:
        pass

    return smiles, thermo_state, dummy_dist

if __name__ == "__main__":
    MOL_DF_PATH = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/mol_df.pkl"
    OUTPUT_PATH = "fiar_pipeline/data/phase4_feature_cache.pt"

    print("[*] Loading unique SMILES...")
    mol_df = pd.read_pickle(MOL_DF_PATH)
    unique_smiles = mol_df['smiles'].dropna().unique()

    print(f"[*] Processing {len(unique_smiles)} molecules...")
    # Use n_jobs=-1 to use all CPU cores
    results = Parallel(n_jobs=4)(
        delayed(_extract_features)(smi) for smi in tqdm(unique_smiles)
    )

    print("[*] Building cache dictionary...")
    cache = {
        smi: {"thermo": thermo, "dist": dist}
        for smi, thermo, dist in results
    }

    torch.save(cache, OUTPUT_PATH)
    print(f"[+] Saved Phase 4 cache to {OUTPUT_PATH}")