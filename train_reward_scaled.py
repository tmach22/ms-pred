"""
Reward-Scaled Teacher Forcing Fine-Tuning — ICEBERG FragGNN
===========================================================
Reads reward_scaled_tensors/ parquets (ms1_intact_smiles, target_ms2_formula,
offline_reward_scalar, collision_energy) and fine-tunes FragGNN to assign
higher atom-leaving probabilities to bond-cuts that yield high-reward fragments.

Batch construction is done on-the-fly via FragmentEngine (no pre-computed
MAGMA trees required). Each sample produces one root-level training example:
  • frag_graph   = root molecule DGL graph
  • targ_atoms   = binary mask over root atoms (pivot atoms whose removal
                   yields a depth-1 fragment matching target_ms2_formula)
  • loss scaling = batch-normalised sigmoid of offline_reward_scalar

Usage:
    cd /data/nas-gpu/wang/tmach007/ms-pred
    python train_reward_scaled.py \\
        --data_dir data/MSnLib/reward_scaled_tensors/ \\
        --ckpt     weights/nist_iceberg_generate.ckpt \\
        --epochs 20 --batch_size 16 --lr 1e-5
"""
from __future__ import annotations

import argparse
import ast
import glob
import os
import sys
import time
from pathlib import Path
from typing import Optional

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_REPO_SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(_REPO_SRC))

from ms_pred.dag_pred.gen_model import FragGNN
from ms_pred.dag_pred.dag_data import TreeProcessor
import ms_pred.common as common
import ms_pred.magma.fragmentation as fragmentation


DEFAULT_ADDUCT = "[M+H]+"
ADDUCT_IDX     = common.ion2onehot_pos[DEFAULT_ADDUCT]   # → 0


def _parse_ce(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, (list, tuple, np.ndarray)):
        flat = [float(x) for x in np.array(val).flatten()]
        return float(np.mean(flat)) if flat else 0.0
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return float(np.mean(parsed)) if parsed else 0.0
        return float(parsed)
    except Exception:
        return 0.0


def _form_dense(formula: str) -> np.ndarray:
    return common.formula_to_dense(formula).astype(np.int64)


class RewardScaledICEBERGDataset(Dataset):
    def __init__(self, parquet_dir: str, tree_processor: TreeProcessor):
        files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet files in {parquet_dir}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        self.data = df.to_dict("records")
        self.tp   = tree_processor
        print(f"[Train] Loaded {len(self.data)} reward-scaled samples.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[dict]:
        row         = self.data[idx]
        parent_smi  = row["ms1_intact_smiles"]
        target_form = row["target_ms2_formula"]
        reward      = float(row["offline_reward_scalar"])
        ce          = _parse_ce(row["collision_energy"])

        try:
            engine = fragmentation.FragmentEngine(mol_str=parent_smi)
            engine.generate_fragments()

            root_frag = engine.get_root_frag()
            root_dict = self.tp.featurize_frag(frag=root_frag, engine=engine)
            root_graph = root_dict["graph"]
            if self.tp.pe_embed_k > 0:
                self.tp.add_pe_embed(root_graph)
            old_to_new = root_dict["old_to_new"]

            # Target formula as dense int vector for robust comparison
            target_dense = _form_dense(target_form)

            # Find pivot atoms (parent_ind_removed) whose removal yields target_form
            pulled_atoms: set[int] = set()
            for entry in engine.frag_to_entry.values():
                if entry["tree_depth"] != 1:
                    continue
                frag_dense = _form_dense(entry.get("form", ""))
                if not np.array_equal(frag_dense, target_dense):
                    continue
                for pa in entry.get("parent_ind_removed", []):
                    pulled_atoms.add(pa)

            if not pulled_atoms:
                return None

            targ_vec = np.zeros(root_graph.num_nodes(), dtype=np.float32)
            for a in pulled_atoms:
                if a < len(old_to_new):
                    new_idx = int(old_to_new[a])
                    if new_idx < root_graph.num_nodes():
                        targ_vec[new_idx] = 1.0

            if targ_vec.sum() == 0:
                return None

            precursor_mz  = float(
                common.mass_from_smi(parent_smi) + common.ion2mass[DEFAULT_ADDUCT]
            )
            root_form     = common.form_from_smi(parent_smi)
            root_form_vec = _form_dense(root_form)

            return {
                "root_repr":     root_graph,
                "frag_graph":    root_graph,        # depth-0: frag IS the root
                "targ":          torch.from_numpy(targ_vec),
                "max_broken":    0,
                "form_vec":      root_form_vec,     # fragment formula = root formula
                "root_form_vec": root_form_vec,
                "adduct":        ADDUCT_IDX,
                "collision_eng": ce,
                "precursor_mz":  precursor_mz,
                "reward":        reward,
            }

        except Exception:
            return None


def reward_collate_fn(raw_batch):
    batch = [b for b in raw_batch if b is not None]
    if not batch:
        return None

    root_reprs  = [b["root_repr"]  for b in batch]
    frag_graphs = [b["frag_graph"] for b in batch]
    targs       = [b["targ"]       for b in batch]

    batched_roots = dgl.batch(root_reprs)
    batched_frags = dgl.batch(frag_graphs)
    targs_padded  = nn.utils.rnn.pad_sequence(targs, batch_first=True)

    frag_atoms  = torch.LongTensor([g.num_nodes() for g in frag_graphs])
    root_inds   = torch.arange(len(batch))          # 1 frag per molecule
    max_broken  = torch.LongTensor([b["max_broken"]    for b in batch])
    adducts     = torch.FloatTensor([b["adduct"]       for b in batch])
    coll_engs   = torch.FloatTensor([b["collision_eng"] for b in batch])
    prec_mzs    = torch.FloatTensor([b["precursor_mz"] for b in batch])
    rewards     = torch.FloatTensor([b["reward"]       for b in batch])
    form_vecs   = torch.LongTensor(np.array([b["form_vec"]      for b in batch]))
    root_vecs   = torch.LongTensor(np.array([b["root_form_vec"] for b in batch]))

    return {
        "names":          [str(i) for i in range(len(batch))],
        "root_reprs":     batched_roots,
        "frag_graphs":    batched_frags,
        "targ_atoms":     targs_padded,
        "frag_atoms":     frag_atoms,
        "inds":           root_inds,
        "broken_bonds":   max_broken,
        "adducts":        adducts,
        "collision_engs": coll_engs,
        "precursor_mzs":  prec_mzs,
        "root_form_vecs": root_vecs,
        "frag_form_vecs": form_vecs,
        "instruments":    None,
        "rewards":        rewards,
    }


def main():
    parser = argparse.ArgumentParser(description="Reward-Scaled ICEBERG Fine-Tuning")
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--epochs",      default=20,   type=int)
    parser.add_argument("--batch_size",  default=16,   type=int)
    parser.add_argument("--lr",          default=1e-5, type=float)
    parser.add_argument("--num_workers", default=4,    type=int)
    args = parser.parse_args()

    os.makedirs("data/MSnLib/logs",      exist_ok=True)
    os.makedirs("weights/reward_scaled", exist_ok=True)

    LOG_PATH = "data/MSnLib/logs/finetune_iceberg.log"
    CKPT_DIR = "weights/reward_scaled"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    log("Loading ICEBERG FragGNN checkpoint...")
    model = FragGNN.load_from_checkpoint(args.ckpt, map_location=device)
    model.train()
    model.to(device)
    log(f"Model loaded. Hidden size: {model.hidden_size}")

    hp = model.hparams
    tree_processor = TreeProcessor(
        pe_embed_k       = int(hp.get("pe_embed_k", 0)),
        root_encode      = hp.get("root_encode", "gnn"),
        add_hs           = bool(hp.get("add_hs", True)),
        embed_elem_group = bool(hp.get("embed_elem_group", False)),
    )

    log(f"Building dataset from {args.data_dir}...")
    dataset = RewardScaledICEBERGDataset(args.data_dir, tree_processor)

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = reward_collate_fn,
        pin_memory  = (device.type == "cuda"),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_loss = float("inf")
    t_start   = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        n_batches    = 0

        for batch in loader:
            if batch is None:
                continue

            rewards = batch.pop("rewards").to(device)

            # Batch-normalised reward weights → sigmoid(z-score) ∈ (0,1)
            # Use population std (correction=0) so single-element batches don't give nan
            r_mean = rewards.mean()
            r_std  = rewards.std(correction=0).clamp(min=1e-6)
            r_weight = torch.sigmoid((rewards - r_mean) / r_std)

            # Move tensors to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)
                elif isinstance(v, dgl.DGLGraph):
                    batch[k] = v.to(device)

            # Forward pass (mirrors _common_step)
            pred_leaving = model.forward(
                batch["frag_graphs"],
                batch["root_reprs"],
                batch["inds"],
                broken         = batch["broken_bonds"],
                adducts        = batch["adducts"],
                collision_engs = batch["collision_engs"],
                precursor_mzs  = batch["precursor_mzs"],
                instruments    = batch["instruments"],
                root_forms     = batch["root_form_vecs"],
                frag_forms     = batch["frag_form_vecs"],
            )

            # Per-sample BCE loss, masked to valid atom positions
            targ_atoms = batch["targ_atoms"].float()
            frag_atoms = batch["frag_atoms"]

            raw_loss = model.bce_loss(pred_leaving, targ_atoms)  # [B, max_atoms]
            is_valid = (
                torch.arange(raw_loss.shape[1], device=device)[None, :]
                < frag_atoms[:, None]
            )
            # Mean loss per sample, weighted by reward, then averaged over batch
            per_sample_loss = (raw_loss * is_valid).sum(dim=1) / frag_atoms.float().clamp(min=1)
            scaled_loss     = (per_sample_loss * r_weight).mean()

            optimizer.zero_grad()
            scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(scaled_loss.item())
            n_batches += 1

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        elapsed  = (time.time() - t_start) / 60
        log(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"Loss: {avg_loss:.6f}  "
            f"Batches: {n_batches}  "
            f"Elapsed: {elapsed:.1f}min"
        )

        ckpt_path = os.path.join(CKPT_DIR, f"reward_scaled_ep{epoch:03d}.ckpt")
        torch.save(
            {"epoch": epoch, "state_dict": model.state_dict(), "loss": avg_loss},
            ckpt_path,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "loss": avg_loss},
                os.path.join(CKPT_DIR, "reward_scaled_best.ckpt"),
            )
            log(f"  ** New best checkpoint saved (loss={best_loss:.6f})")

    log(f"Training complete. Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
