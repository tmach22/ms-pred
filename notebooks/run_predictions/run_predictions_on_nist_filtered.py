import pandas as pd
import numpy as np
import os
import sys
import torch
from tqdm.notebook import tqdm
import time

# --- 1. SETUP MS-PRED & ICEBERG ---
repo_root = '/data/nas-gpu/wang/tmach007/ms-pred'
repo_src = os.path.join(repo_root, 'src')
if repo_src not in sys.path: sys.path.append(repo_src)

try:
    import ms_pred.common as common
    from ms_pred.dag_pred.iceberg_elucidation import (
        iceberg_prediction, 
        load_global_config, 
        load_pred_spec
    )
    print("ICEBERG modules loaded successfully.")
except ImportError:
    print("Error: Could not find ms_pred. Ensure the path is correct.")

# --- 2. CONFIGURATION ---
INPUT_FEATHER = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/data_splits/stratified_binary_07_unseen_nist.feather"
SPEC_LOOKUP = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/spec_df.pkl"
MOL_LOOKUP = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/mol_df.pkl"
ICEBERG_YAML = os.path.join(repo_root, 'configs/iceberg/iceberg_elucidation.yaml')

OUTPUT_PRED_SPEC = "./results/msg_iceberg_predictions/iceberg_predicted_spec_df.pkl"
os.makedirs(os.path.dirname(OUTPUT_PRED_SPEC), exist_ok=True)

# --- 3. HELPERS ---
def clean_adduct(adduct):
    if pd.isna(adduct) or not adduct: return '[M+H]+'
    s = str(adduct).strip().replace('1+', '+').replace('1-', '-').replace('NA', 'Na').replace('K+', 'K')
    if not s.startswith('['): s = f"[{s}]"
    return s

def get_iceberg_peaks(smiles, nce, adduct, local_config):
    """Generates peaks for a single molecule under specific conditions."""
    try:
        os.chdir(repo_root)
        res_path, pmz = iceberg_prediction(
            candidate_smiles=[smiles], 
            collision_energies=[nce], 
            adduct=adduct, 
            nce=True,
            force_recompute=False, 
            **local_config
        )
        _, pred_specs, _ = load_pred_spec(load_dir=res_path, merge_spec=False)
        
        if len(pred_specs) > 0:
            spec_dict = pred_specs[0]
            # Key is typically stringified float of NCE
            target_key = list(spec_dict.keys())[0]
            raw_peaks = spec_dict[target_key]
            filtered_peaks = [(float(p[0]), float(p[1])) for p in raw_peaks if p[1] > 0.005]
            return filtered_peaks, pmz
    except Exception as e:
        print(f"Exception while predicting the spectrum: {e}")
        return None, None
    finally:
        os.chdir(original_cwd)
    return None, None

# --- 4. PREPARE METADATA ---
print("Loading test set and metadata...")
test_df = pd.read_feather(INPUT_FEATHER)
# Get all unique spec_ids involved in the test set
unique_sids = pd.unique(test_df[['name_main', 'name_sub']].values.ravel('K'))

spec_df = pd.read_pickle(SPEC_LOOKUP)
mol_df = pd.read_pickle(MOL_LOOKUP)

# Create a master metadata table for just our test IDs
meta_df = spec_df[spec_df['spec_id'].isin(unique_sids)].merge(
    mol_df[['mol_id', 'smiles']], on='mol_id', how='left'
)
meta_lookup = meta_df.set_index('spec_id')

# --- 5. EXECUTION ---
original_cwd = os.getcwd()
local_config = load_global_config(ICEBERG_YAML)
local_config.update({'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'})

predicted_library = []

print(f"Generating predictions for {len(unique_sids)} unique spectra...")

running_avg = 0

for sid in tqdm(unique_sids, desc="Building In Silico Library"):
    m = meta_lookup.loc[sid]
    
    # Use the experimental parameters directly from the lookup
    nce = m['nce_updated'] if not pd.isna(m['nce_updated']) else (m['nce'] if not pd.isna(m['nce']) else 30.0)
    adduct = clean_adduct(m['prec_type'])
    # print(f"NCE: {nce} | Adduct: {adduct} | SMILES: {m['smiles']}")

    start_time = time.time()
    
    peaks, pmz = get_iceberg_peaks(m['smiles'], nce, adduct, local_config)

    end_time = time.time()
    
    if peaks:
        record = m.to_dict()
        record['peaks'] = peaks
        record['prec_mz'] = pmz
        predicted_library.append(record)
        print(f"Completed {len(predicted_library)} predictions out of {len(unique_sids)} total.")
        running_avg = (running_avg + (end_time - start_time))/len(predicted_library)
        print(f"{running_avg}s/it  Est Time: {running_avg*(len(unique_sids) - len(predicted_library))} seconds")
        

# --- 6. SAVE ---
pred_spec_df = pd.DataFrame(predicted_library)
if 'smiles' in pred_spec_df.columns:
    pred_spec_df = pred_spec_df.drop(columns=['smiles'])

pred_spec_df.to_pickle(OUTPUT_PRED_SPEC)
print(f"\nSuccess! Predicted library saved to {OUTPUT_PRED_SPEC}")