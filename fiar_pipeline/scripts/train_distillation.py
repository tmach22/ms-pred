import os
import multiprocessing as mp
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fiar_pipeline.data_loaders.distillation_dataloader import (
    MS3DistillationDataset,
    collate_actor_oracle,
)
from fiar_pipeline.extractors.iceberg.scalpel import ICEBERGScalpel
from fiar_pipeline.magma.scorer import magma_score_fragments

_ICEBERG_VALID_ELEMENTS = frozenset([
    "C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F", "Br", "B",
    "Se", "Fe", "Co", "As", "Na", "K",
])

CKPT_PATH         = os.environ.get("CKPT_PATH",         "/home/user/ms-pred/weights/nist_iceberg_generate.ckpt")
TRAIN_PATH        = os.environ.get("TRAIN_PATH",        "/home/user/ms-pred/data/MSnLib/splits_v2/train.parquet")
CHECKPOINT_DIR    = os.environ.get("CHECKPOINT_DIR",    "/home/user/ms-pred/weights/ms3_reranker/")
TOP_K             = 50
BATCH_SIZE        = int(os.environ.get("BATCH_SIZE",        "128"))
MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", "0")) or None
NUM_EPOCHS        = 20
LR                = 1e-3
CPU_WORKERS       = int(os.environ.get("CPU_WORKERS",   "20"))
SAVE_EVERY_STEPS  = int(os.environ.get("SAVE_EVERY_STEPS", "200"))
MAGMA_TIMEOUT     = int(os.environ.get("MAGMA_TIMEOUT", "300"))  # seconds per batch


def make_pool() -> mp.Pool:
    # maxtasksperchild recycles workers periodically to prevent memory growth
    return mp.Pool(processes=CPU_WORKERS, maxtasksperchild=200)


def safe_starmap(pool: mp.Pool, fn, args_list: list) -> Tuple[list, mp.Pool]:
    """starmap with per-batch timeout. Restarts the pool and returns zeros on hang."""
    try:
        return pool.starmap_async(fn, args_list).get(timeout=MAGMA_TIMEOUT), pool
    except mp.TimeoutError:
        print(
            f"[WARN] MAGMa batch timed out after {MAGMA_TIMEOUT}s — restarting pool, zeroing batch rewards",
            flush=True,
        )
        pool.terminate()
        return [0.0] * len(args_list), make_pool()


class MS3ReRanker(nn.Module):
    INPUT_DIM = 13

    def __init__(self, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = self.INPUT_DIM
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_reranker_features(frag_lists: list, ces: torch.Tensor, ms2_mzs: torch.Tensor) -> torch.Tensor:
    batch_feats = []
    for frags, ce_val, mz_val in zip(frag_lists, ces.tolist(), ms2_mzs.tolist()):
        if not frags:
            batch_feats.append(torch.zeros(MS3ReRanker.INPUT_DIM))
            continue

        pg   = torch.tensor([f.prob_gen for f in frags], dtype=torch.float32)
        mass = torch.tensor([f.exact_mass for f in frags], dtype=torch.float32)
        brok = torch.tensor([float(f.max_broken) for f in frags], dtype=torch.float32)
        dep  = torch.tensor([float(f.tree_depth) for f in frags], dtype=torch.float32)

        feat = torch.stack([
            pg.mean(), pg.max(), pg.std(unbiased=False),
            mass.mean() / 500.0, mass.min() / 500.0, mass.max() / 500.0,
            brok.mean() / 6.0, brok.max() / 6.0,
            dep.mean() / 3.0, dep.max() / 3.0,
            torch.tensor(len(frags) / float(TOP_K)),
            torch.tensor(ce_val / 100.0),
            torch.tensor(mz_val / 1000.0),
        ])
        batch_feats.append(feat)

    return torch.stack(batch_feats)


def _is_iceberg_compatible(smiles: str) -> bool:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return False
    return all(atom.GetSymbol() in _ICEBERG_VALID_ELEMENTS for atom in mol.GetAtoms())


def train_epoch(
    epoch: int,
    scalpel: ICEBERGScalpel,
    reranker: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cpu_pool: mp.Pool,
    device: torch.device,
    checkpoint_dir: str = None,
    save_every_steps: int = 200,
) -> Tuple[float, float, mp.Pool]:
    reranker.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_reward = 0.0

    for step, batch in enumerate(dataloader):
        if batch is None:
            continue

        smiles_list, batched_ces, batched_ms2_mz, padded_ms3_mz, padded_ms3_int = batch
        batched_ces    = batched_ces.to(device)
        batched_ms2_mz = batched_ms2_mz.to(device)

        ces_list = batched_ces.cpu().tolist()
        mzs_list = batched_ms2_mz.cpu().tolist()

        valid_mask    = [_is_iceberg_compatible(s) for s in smiles_list]
        valid_indices = [i for i, v in enumerate(valid_mask) if v]
        valid_smiles  = [smiles_list[i] for i in valid_indices]
        valid_ces     = [ces_list[i]    for i in valid_indices]
        valid_mzs     = [mzs_list[i]   for i in valid_indices]

        with torch.no_grad():
            valid_frags = scalpel.extract_batch(
                smiles_list=valid_smiles, collision_engs=valid_ces, precursor_mzs=valid_mzs
            ) if valid_smiles else []

        frag_iter      = iter(valid_frags)
        frag_lists     = [next(frag_iter) if v else [] for v in valid_mask]
        candidate_smiles = [[fr.smiles for fr in frags if fr.smiles] for frags in frag_lists]

        mz_np  = padded_ms3_mz.numpy()
        int_np = padded_ms3_int.numpy()

        scoring_args = [(candidate_smiles[i], mz_np[i], int_np[i], mzs_list[i]) for i in range(len(smiles_list))]
        rewards, cpu_pool = safe_starmap(cpu_pool, magma_score_fragments, scoring_args)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)

        feat_matrix      = build_reranker_features(frag_lists, batched_ces.cpu(), batched_ms2_mz.cpu()).to(device)
        predicted_scores = reranker(feat_matrix)
        loss             = criterion(predicted_scores, rewards_tensor)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reranker.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss   += loss.item()
        total_reward += rewards_tensor.mean().item()

        if step % 10 == 0:
            print(
                f"Epoch {epoch:02d} | Step {step:04d} | MSE Loss: {loss.item():.4f} | "
                f"Avg MAGMa: {rewards_tensor.mean().item():.4f} | Pred mean: {predicted_scores.mean().item():.4f}",
                flush=True,
            )

        if checkpoint_dir and save_every_steps and (step + 1) % save_every_steps == 0:
            step_path = os.path.join(checkpoint_dir, f"reranker_epoch{epoch:02d}_step{step+1:05d}.pt")
            torch.save(reranker.state_dict(), step_path)
            print(f"[CKPT] Step checkpoint: {step_path}", flush=True)

    return total_loss, total_reward, cpu_pool


def evaluate_epoch(
    epoch: int,
    scalpel: ICEBERGScalpel,
    reranker: nn.Module,
    dataloader: DataLoader,
    cpu_pool: mp.Pool,
    device: torch.device,
) -> Tuple[float, float, mp.Pool]:
    reranker.eval()
    criterion    = nn.MSELoss()
    total_loss   = 0.0
    total_reward = 0.0

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if batch is None:
                continue

            smiles_list, batched_ces, batched_ms2_mz, padded_ms3_mz, padded_ms3_int = batch
            batched_ces    = batched_ces.to(device)
            batched_ms2_mz = batched_ms2_mz.to(device)

            ces_list = batched_ces.cpu().tolist()
            mzs_list = batched_ms2_mz.cpu().tolist()

            valid_mask    = [_is_iceberg_compatible(s) for s in smiles_list]
            valid_indices = [i for i, v in enumerate(valid_mask) if v]
            valid_smiles  = [smiles_list[i] for i in valid_indices]
            valid_ces     = [ces_list[i]    for i in valid_indices]
            valid_mzs     = [mzs_list[i]   for i in valid_indices]

            valid_frags = scalpel.extract_batch(
                smiles_list=valid_smiles, collision_engs=valid_ces, precursor_mzs=valid_mzs
            ) if valid_smiles else []

            frag_iter        = iter(valid_frags)
            frag_lists       = [next(frag_iter) if v else [] for v in valid_mask]
            candidate_smiles = [[fr.smiles for fr in frags if fr.smiles] for frags in frag_lists]

            mz_np  = padded_ms3_mz.numpy()
            int_np = padded_ms3_int.numpy()

            scoring_args = [
                (candidate_smiles[i], mz_np[i], int_np[i], mzs_list[i])
                for i in range(len(smiles_list))
            ]
            rewards, cpu_pool = safe_starmap(cpu_pool, magma_score_fragments, scoring_args)
            rewards_tensor    = torch.tensor(rewards, dtype=torch.float32, device=device)

            feat_matrix      = build_reranker_features(frag_lists, batched_ces.cpu(), batched_ms2_mz.cpu()).to(device)
            predicted_scores = reranker(feat_matrix)
            loss             = criterion(predicted_scores, rewards_tensor)

            total_loss   += loss.item()
            total_reward += rewards_tensor.mean().item()

    return total_loss, total_reward, cpu_pool


def main():
    device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_pool = make_pool()

    print("=" * 60)
    print("Frozen Generator + Learned Scorer — Single GPU")
    print(f"  Device         : {device}")
    print(f"  ICEBERG ckpt   : {CKPT_PATH}")
    print(f"  Batch size     : {BATCH_SIZE}")
    print(f"  Max train samp : {MAX_TRAIN_SAMPLES or 'all'}")
    print(f"  CPU workers    : {CPU_WORKERS}")
    print(f"  MAGMa timeout  : {MAGMA_TIMEOUT}s/batch")
    print(f"  Checkpoints    : {CHECKPOINT_DIR}")
    print(f"  Save every     : {SAVE_EVERY_STEPS} steps")
    print("=" * 60, flush=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_loss = float('inf')

    VAL_PATH = os.environ.get("VAL_PATH", "/home/user/ms-pred/data/MSnLib/splits_v2/val.parquet")

    train_dataset = MS3DistillationDataset(TRAIN_PATH, max_samples=MAX_TRAIN_SAMPLES)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_actor_oracle,
        num_workers=4,
        pin_memory=True,
    )

    val_dataset = MS3DistillationDataset(VAL_PATH)
    val_loader  = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_actor_oracle,
        num_workers=4,
        pin_memory=True,
    )

    scalpel   = ICEBERGScalpel(ckpt_path=CKPT_PATH, device=str(device), top_k=TOP_K, threshold=0.0, compute_morgan_fp=False)
    reranker  = MS3ReRanker(hidden_dims=[64, 32]).to(device)
    optimizer = torch.optim.Adam(reranker.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_reward, cpu_pool = train_epoch(
            epoch, scalpel, reranker, train_loader, optimizer, cpu_pool, device,
            checkpoint_dir=CHECKPOINT_DIR, save_every_steps=SAVE_EVERY_STEPS,
        )
        avg_train_loss   = train_loss   / max(len(train_loader), 1)
        avg_train_reward = train_reward / max(len(train_loader), 1)

        val_loss, val_reward, cpu_pool = evaluate_epoch(
            epoch, scalpel, reranker, val_loader, cpu_pool, device,
        )
        avg_val_loss   = val_loss   / max(len(val_loader), 1)
        avg_val_reward = val_reward / max(len(val_loader), 1)

        print(f"\n>>> Epoch {epoch:02d} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f} | Val MAGMa: {avg_val_reward:.4f}")

        latest_path = os.path.join(CHECKPOINT_DIR, "reranker_latest.pt")
        torch.save(reranker.state_dict(), latest_path)

        if avg_val_loss < best_loss:
            best_loss  = avg_val_loss
            best_path  = os.path.join(CHECKPOINT_DIR, "reranker_best.pt")
            torch.save(reranker.state_dict(), best_path)
            print(f"[BEST] Epoch {epoch} | Val Loss: {best_loss:.4f}", flush=True)

    cpu_pool.close()
    cpu_pool.join()


if __name__ == "__main__":
    main()
