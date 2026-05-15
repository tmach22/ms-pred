import os
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fiar_pipeline.data_loaders.distillation_dataloader import MS3DistillationDataset, collate_actor_oracle
from fiar_pipeline.extractors.iceberg.scalpel import ICEBERGScalpel
from fiar_pipeline.magma.scorer import magma_score_fragments
from fiar_pipeline.scripts.train_distillation import _is_iceberg_compatible


def _score_molecule(args):
    """Module-level worker: scores all candidates for one molecule, returns list of floats.
    Must be at module level so multiprocessing can pickle it.
    Receives the spectrum arrays once instead of once-per-candidate, eliminating 50x IPC redundancy."""
    candidate_list, mz_arr, int_arr, precursor_mz = args
    return [magma_score_fragments([cand], mz_arr, int_arr, precursor_mz) for cand in candidate_list]


def apply_temperature_softmax(raw_scores: list, temperature: float) -> list:
    if not raw_scores:
        return []
    scores_tensor = torch.tensor(raw_scores, dtype=torch.float32)
    soft_labels = F.softmax(scores_tensor / temperature, dim=0)
    return soft_labels.tolist()


def process_batch(batch, scalpel, cpu_pool, device, temperature):
    if batch is None:
        return []

    smiles_list, batched_ces, batched_ms2_mz, padded_ms3_mz, padded_ms3_int = batch
    ces_list = batched_ces.tolist()
    mzs_list = batched_ms2_mz.tolist()

    valid_mask    = [_is_iceberg_compatible(s) for s in smiles_list]
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    valid_smiles  = [smiles_list[i] for i in valid_indices]
    valid_ces     = [ces_list[i]    for i in valid_indices]
    valid_mzs     = [mzs_list[i]   for i in valid_indices]

    with torch.no_grad():
        valid_frags = scalpel.extract_batch(valid_smiles, valid_ces, valid_mzs) if valid_smiles else []

    frag_iter  = iter(valid_frags)
    frag_lists = [next(frag_iter) if v else [] for v in valid_mask]

    candidate_smiles = [[fr.smiles for fr in frags if fr.smiles] for frags in frag_lists]

    mz_np  = padded_ms3_mz.numpy()
    int_np = padded_ms3_int.numpy()

    valid_mol_indices = [i for i in range(len(smiles_list)) if candidate_smiles[i]]
    scoring_args = [
        (candidate_smiles[i], mz_np[i], int_np[i], mzs_list[i])
        for i in valid_mol_indices
    ]

    if scoring_args:
        mol_score_lists = cpu_pool.map(_score_molecule, scoring_args)
    else:
        mol_score_lists = []

    batch_results = []
    for i, raw_scores in zip(valid_mol_indices, mol_score_lists):
        if not raw_scores:
            continue
        soft_labels = apply_temperature_softmax(raw_scores, temperature)
        batch_results.append({
            'intact_parent_smiles': smiles_list[i],
            'collision_energy':     ces_list[i],
            'ms2_precursor_mz':     mzs_list[i],
            'candidate_fragments':  candidate_smiles[i],
            'raw_magma_scores':     raw_scores,
            'soft_labels':          soft_labels,
        })

    return batch_results


def main():
    # Paths relative to the PVC mount point at /workspace
    TRAIN_PATH  = "/workspace/data/train.parquet"
    CKPT_PATH   = "/workspace/weights/nist_iceberg_generate.ckpt"

    TEMPERATURE = 0.1
    BATCH_SIZE  = 128
    CPU_WORKERS = int(os.environ.get("CPU_WORKERS", "20"))

    # Indexed Job sharding — K8s injects JOB_COMPLETION_INDEX (0-based).
    # NUM_SHARDS must match spec.completions in the Job YAML.
    SHARD_INDEX = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    NUM_SHARDS  = int(os.environ.get("NUM_SHARDS", "1"))

    OUTPUT_PATH = f"/workspace/results/train_soft_labels_shard_{SHARD_INDEX:02d}.parquet"
    os.makedirs("/workspace/results", exist_ok=True)

    device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_pool = mp.Pool(processes=CPU_WORKERS)

    print("=" * 60)
    print("Soft-Label Generation Pipeline (NRP / Nautilus)")
    print(f"  Shard       : {SHARD_INDEX + 1} / {NUM_SHARDS}")
    print(f"  Hardware    : {CPU_WORKERS} CPU Cores | {device}")
    print(f"  Temperature : {TEMPERATURE}")
    print("=" * 60)

    full_dataset = MS3DistillationDataset(TRAIN_PATH)
    total        = len(full_dataset)
    shard_size   = (total + NUM_SHARDS - 1) // NUM_SHARDS
    start        = SHARD_INDEX * shard_size
    end          = min(start + shard_size, total)

    from torch.utils.data import Subset
    dataset = Subset(full_dataset, range(start, end))
    print(f"  Rows        : {start}–{end} ({end - start} molecules)")

    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        collate_fn  = collate_actor_oracle,
        num_workers = 4,
    )

    scalpel     = ICEBERGScalpel(ckpt_path=CKPT_PATH, device=str(device), top_k=50)
    all_results = []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluating Candidates")):
        if MAX_BATCHES and step >= MAX_BATCHES:
            print(f"\nReached debug limit of {MAX_BATCHES} batches. Stopping generation.")
            break

        batch_results = process_batch(batch, scalpel, cpu_pool, device, TEMPERATURE)
        all_results.extend(batch_results)

    df_final = pd.DataFrame(all_results)
    df_final.to_parquet(OUTPUT_PATH)
    print(f"\n✅ Saved {len(df_final)} processed molecules to {OUTPUT_PATH}")

    cpu_pool.close()
    cpu_pool.join()


if __name__ == "__main__":
    main()
