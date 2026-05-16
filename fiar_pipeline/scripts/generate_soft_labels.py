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

    # One task per molecule: each worker receives the spectrum once and scores all candidates
    # internally. Cuts IPC serialization from ~6400 round-trips to 128 per batch.
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
    import argparse
    parser = argparse.ArgumentParser(description="Soft-Label Generation Pipeline (Sharded)")
    parser.add_argument("--input",      type=str,   required=True,       help="Path to input parquet shard")
    parser.add_argument("--output",     type=str,   required=True,       help="Path to output parquet shard")
    parser.add_argument("--device",     type=str,   required=True,       help="Target GPU (e.g., cuda:0 or cuda:1)")
    parser.add_argument("--workers",    type=int,   default=14,          help="Number of CPU workers for MAGMa")
    parser.add_argument("--max_batches",type=int,   default=None,        help="Debug batch limit (None for all)")
    parser.add_argument("--temp",       type=float, default=0.1,         help="Softmax temperature")
    args = parser.parse_args()

    CKPT_PATH  = "/data/nas-gpu/wang/tmach007/ms-pred/weights/nist_iceberg_generate.ckpt"
    BATCH_SIZE = 128

    device   = torch.device(args.device)
    torch.cuda.set_device(device)  # required for sparse tensor ops on non-primary GPU
    cpu_pool = mp.Pool(processes=args.workers)

    print("=" * 60)
    print(f"Soft-Label Pipeline | Target: {args.device}")
    print(f"  Input       : {args.input}")
    print(f"  Output      : {args.output}")
    print(f"  CPU Workers : {args.workers}")
    print(f"  Temperature : {args.temp}")
    print("=" * 60)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    dataset    = MS3DistillationDataset(args.input)
    dataloader = DataLoader(
        dataset,
        batch_size = BATCH_SIZE,
        shuffle    = False,
        collate_fn = collate_actor_oracle,
        num_workers = 4,
    )

    scalpel     = ICEBERGScalpel(ckpt_path=CKPT_PATH, device=str(device), top_k=50)
    all_results = []

    for step, batch in enumerate(tqdm(dataloader, desc=f"Evaluating [{args.device}]")):
        if args.max_batches and step >= args.max_batches:
            print(f"\nReached debug limit of {args.max_batches} batches. Stopping.")
            break

        batch_results = process_batch(batch, scalpel, cpu_pool, device, args.temp)
        all_results.extend(batch_results)

    df_final = pd.DataFrame(all_results)
    df_final.to_parquet(args.output)
    print(f"\n✅ Saved {len(df_final)} processed molecules to {args.output}")

    cpu_pool.close()
    cpu_pool.join()


if __name__ == "__main__":
    main()
