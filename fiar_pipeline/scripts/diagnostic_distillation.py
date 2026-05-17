"""
Diagnostic script: runs 3 train batches -> 3 val batches -> checkpoint save.
Exercises the full train_epoch/evaluate_epoch/checkpoint flow in ~5 minutes
so you can confirm the fragmentation.py None-guard fix works before
relaunching the full 20-epoch training run.

Run from ms-pred/:
    python -m fiar_pipeline.scripts.diagnostic_distillation
"""
import os
import multiprocessing as mp
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fiar_pipeline.data_loaders.distillation_dataloader import (
    MS3DistillationDataset,
    collate_actor_oracle,
)
from fiar_pipeline.extractors.iceberg.scalpel import ICEBERGScalpel
from fiar_pipeline.magma.scorer import magma_score_fragments
from fiar_pipeline.scripts.train_distillation import (
    MS3ReRanker,
    build_reranker_features,
    train_epoch,
    evaluate_epoch,
    _is_iceberg_compatible,
    CKPT_PATH,
    TOP_K,
)

TRAIN_PATH = os.environ.get("TRAIN_PATH", "/home/user/ms-pred/data/MSnLib/splits_v2/train.parquet")
VAL_PATH   = os.environ.get("VAL_PATH",   "/home/user/ms-pred/data/MSnLib/splits_v2/val.parquet")
DIAG_DIR   = os.environ.get("DIAG_DIR",   "/home/user/ms-pred/weights/ms3_reranker_diag/")

BATCH_SIZE  = 32   # small for speed
CPU_WORKERS = 4
N_BATCHES   = 3    # batches to run in train and val


def make_loader(path: str, n_batches: int, shuffle: bool) -> DataLoader:
    ds = MS3DistillationDataset(path)
    subset = Subset(ds, range(min(n_batches * BATCH_SIZE, len(ds))))
    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate_actor_oracle,
        num_workers=2,
        pin_memory=False,
    )


def main():
    device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_pool = mp.Pool(processes=CPU_WORKERS)

    print("=" * 60)
    print("Distillation Diagnostic — 3 train + 3 val batches")
    print(f"  Device      : {device}")
    print(f"  BATCH_SIZE  : {BATCH_SIZE}")
    print(f"  CPU_WORKERS : {CPU_WORKERS}")
    print("=" * 60, flush=True)

    os.makedirs(DIAG_DIR, exist_ok=True)

    train_loader = make_loader(TRAIN_PATH, N_BATCHES, shuffle=True)
    val_loader   = make_loader(VAL_PATH,   N_BATCHES, shuffle=False)

    scalpel  = ICEBERGScalpel(ckpt_path=CKPT_PATH, device=str(device), top_k=TOP_K, threshold=0.0, compute_morgan_fp=False)
    reranker = MS3ReRanker(hidden_dims=[64, 32]).to(device)
    optimizer = torch.optim.Adam(reranker.parameters(), lr=1e-3)

    print("\n[1/3] Running train_epoch ...", flush=True)
    train_loss, train_reward = train_epoch(
        epoch=1, scalpel=scalpel, reranker=reranker,
        dataloader=train_loader, optimizer=optimizer,
        cpu_pool=cpu_pool, device=device,
    )
    avg_train_loss = train_loss / max(len(train_loader), 1)
    print(f"  Train MSE : {avg_train_loss:.4f}  (raw={train_loss:.4f})", flush=True)

    print("\n[2/3] Running evaluate_epoch on val set ...", flush=True)
    val_loss, val_reward = evaluate_epoch(
        epoch=1, scalpel=scalpel, reranker=reranker,
        dataloader=val_loader, cpu_pool=cpu_pool, device=device,
    )
    avg_val_loss = val_loss / max(len(val_loader), 1)
    print(f"  Val MSE   : {avg_val_loss:.4f}  (raw={val_loss:.4f})", flush=True)

    print("\n[3/3] Saving checkpoints ...", flush=True)
    latest_path = os.path.join(DIAG_DIR, "reranker_latest.pt")
    best_path   = os.path.join(DIAG_DIR, "reranker_best.pt")
    torch.save(reranker.state_dict(), latest_path)
    torch.save(reranker.state_dict(), best_path)
    assert os.path.exists(latest_path) and os.path.getsize(latest_path) > 0
    assert os.path.exists(best_path)   and os.path.getsize(best_path)   > 0
    print(f"  Saved: {latest_path}")
    print(f"  Saved: {best_path}")

    cpu_pool.close()
    cpu_pool.join()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC PASSED — all three stages completed without error.")
    print("=" * 60)


if __name__ == "__main__":
    main()
