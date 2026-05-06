"""
ICEBERG-Generate MS³ Fine-Tuner
================================
Fine-tunes the FragGNN backbone (ICEBERG-Generate) using empirical MS³ edges
from MSnLib.  A frozen oracle copy of the model provides a KL regularisation
signal to prevent catastrophic forgetting of the MS2 fragmentation prior.

Architecture
------------
  Active model (π_θ):  FragGNN loaded from checkpoint, output_map unfrozen.
  Oracle model (π₀):   Identical weights, fully frozen, used for KL penalty.
  KineticFiLM:         Learnable NCE gate injected at the AvgPooling step via
                       a forward hook on active_model.pool.

Loss
----
  L = w_gs * GaussianSoftTargetBCE
    + w_pu * MaskedPUMarginLoss
    + w_kl * KLDistillationLoss

Training protocol
-----------------
  - Backbone GNN frozen; only output_map + KineticFiLM params are updated.
  - AdamW, lr = 1e-4, weight_decay = 1e-5.
  - Cosine annealing LR schedule.
  - Early stopping on validation GaussianSoftTargetBCE (patience = 5).
  - Checkpoint saved on every validation improvement.

Usage
-----
cd /data/nas-gpu/wang/tmach007/ms-pred

/data/nas-gpu/wang/tmach007/ms-pred/y/envs/ms-gen/bin/python \\
    fiar_pipeline/train/finetune_iceberg_ms3.py \\
    --feather       data/ms3_edge_labels.feather \\
    --ckpt_path     weights/nist_iceberg_generate.ckpt \\
    --output_dir    fiar_pipeline/results/iceberg_ms3 \\
    --epochs        30 \\
    --batch_size    64 \\
    --lr            1e-4 \\
    --w_gs          1.0 \\
    --w_pu          0.5 \\
    --w_kl          0.3 \\
    --device        cuda:0
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ── Repo path injection ────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import ms_pred.common as common
from ms_pred.dag_pred.gen_model import FragGNN

from fiar_pipeline.data_loaders.ms3_edge_dataset import MS3EdgeDataset
from fiar_pipeline.model.kinetic_film import KineticFiLM
from fiar_pipeline.model.ms3_losses import (
    GaussianSoftTargetBCE,
    MaskedPUMarginLoss,
    KLDistillationLoss,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_frag_gnn(ckpt_path: str, device: torch.device) -> FragGNN:
    """Load a FragGNN from a PyTorch-Lightning checkpoint (.ckpt)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    model = FragGNN(**hparams)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    return model


def freeze_backbone(model: FragGNN) -> None:
    """Freeze all parameters except output_map."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("output_map.")


# ── KineticFiLM forward-hook injection ────────────────────────────────────────

class _FiLMContext:
    """Mutable container passed into the pool forward hook closure."""
    nce_per_frag: Optional[torch.Tensor] = None
    film: Optional[KineticFiLM] = None


_FILM_CTX = _FiLMContext()


def _pool_hook(module: nn.Module, _input, output: torch.Tensor) -> torch.Tensor:
    """Forward hook on FragGNN.pool — applies KineticFiLM in-place."""
    if _FILM_CTX.film is None or _FILM_CTX.nce_per_frag is None:
        return output
    return _FILM_CTX.film(output, _FILM_CTX.nce_per_frag)


def _set_nce(batch: dict, ind_maps: torch.Tensor, device: torch.device) -> None:
    """Populate _FILM_CTX.nce_per_frag from per-molecule collision_engs."""
    # collision_engs: [n_mols]  →  nce: [n_frags]
    collision_engs = batch["collision_engs"].to(device)          # [n_mols]
    nce_per_mol = collision_engs / 50.0                          # → NCE
    _FILM_CTX.nce_per_frag = nce_per_mol[ind_maps.to(device)]    # [n_frags]


# ── Forward helper ─────────────────────────────────────────────────────────────

def _model_forward(model: FragGNN, batch: dict, device: torch.device) -> torch.Tensor:
    """Run model forward and return padded sigmoid outputs [batch, max_atoms]."""
    return model(
        graphs         = batch["frag_graphs"].to(device),
        root_repr      = batch["root_reprs"].to(device),
        ind_maps       = batch["inds"].to(device),
        broken         = batch["broken_bonds"].to(device),
        collision_engs = batch["collision_engs"].to(device),
        precursor_mzs  = batch["precursor_mzs"].to(device),
        adducts        = batch["adducts"].to(device) if batch.get("adducts") is not None else None,
        instruments    = batch["instruments"].to(device) if batch.get("instruments") is not None else None,
        root_forms     = batch["root_form_vecs"].to(device),
        frag_forms     = batch["frag_form_vecs"].to(device),
    )  # [n_frags, max_atoms]


# ── Train / Val loops ──────────────────────────────────────────────────────────

def _run_epoch(
    active: FragGNN,
    oracle: FragGNN,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    loss_gs: GaussianSoftTargetBCE,
    loss_pu: MaskedPUMarginLoss,
    loss_kl: KLDistillationLoss,
    w_gs: float,
    w_pu: float,
    w_kl: float,
    is_train: bool,
) -> dict[str, float]:
    active.train(is_train)
    # Keep BN stats frozen in the backbone; only output_map + film in train mode
    if is_train:
        active.output_map.train()

    total_loss = total_gs = total_pu = total_kl = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch in loader:
            if batch is None:
                continue

            targ = batch["targ_atoms"].to(device)           # [n_frags, max_atoms]
            natoms = batch["frag_atoms"].to(device)         # [n_frags]

            # Prepare FiLM context before active forward pass
            _set_nce(batch, batch["inds"], device)

            pred_active = _model_forward(active, batch, device)  # [n_frags, max_atoms]

            # Oracle forward (always no_grad)
            with torch.no_grad():
                pred_oracle = _model_forward(oracle, batch, device)

            gs = loss_gs(pred_active, targ, natoms)
            pu = loss_pu(pred_active, targ, natoms)
            kl = loss_kl(pred_active, pred_oracle, natoms)
            loss = w_gs * gs + w_pu * pu + w_kl * kl

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in active.parameters() if p.requires_grad]
                    + list(_FILM_CTX.film.parameters()),
                    max_norm=5.0,
                )
                optimizer.step()

            total_loss += loss.item()
            total_gs   += gs.item()
            total_pu   += pu.item()
            total_kl   += kl.item()
            n_batches  += 1

    denom = max(n_batches, 1)
    return {
        "loss": total_loss / denom,
        "gs":   total_gs   / denom,
        "pu":   total_pu   / denom,
        "kl":   total_kl   / denom,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"[*] Loading dataset: {args.feather}")
    full_dataset = MS3EdgeDataset(
        feather_path = args.feather,
        max_mol_size = args.max_mol_size,
        ion_mode     = args.ion_mode,
    )
    print(f"    {len(full_dataset):,} edges after size filter")

    val_n   = max(1, int(len(full_dataset) * args.val_frac))
    train_n = len(full_dataset) - val_n
    train_ds, val_ds = random_split(full_dataset, [train_n, val_n])

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        collate_fn  = MS3EdgeDataset.collate_fn,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size * 2,
        shuffle     = False,
        collate_fn  = MS3EdgeDataset.collate_fn,
        num_workers = args.num_workers,
        pin_memory  = True,
    )

    # ── Models ────────────────────────────────────────────────────────────────
    print(f"[*] Loading FragGNN checkpoint: {args.ckpt_path}")
    active_model = load_frag_gnn(args.ckpt_path, device).to(device)
    oracle_model = copy.deepcopy(active_model).to(device)

    # Freeze oracle entirely
    for p in oracle_model.parameters():
        p.requires_grad = False
    oracle_model.eval()

    # Freeze active backbone; only output_map trains
    freeze_backbone(active_model)
    trainable_backbone = [p for p in active_model.parameters() if p.requires_grad]
    print(f"    Backbone trainable params: {sum(p.numel() for p in trainable_backbone):,}")

    # KineticFiLM
    film = KineticFiLM(hidden_size=active_model.hidden_size).to(device)
    _FILM_CTX.film = film
    hook_handle = active_model.pool.register_forward_hook(_pool_hook)
    print(f"    KineticFiLM params: {sum(p.numel() for p in film.parameters()):,}")

    # ── Losses ────────────────────────────────────────────────────────────────
    loss_gs = GaussianSoftTargetBCE()
    loss_pu = MaskedPUMarginLoss(margin=args.pu_margin)
    loss_kl = KLDistillationLoss(temperature=args.kl_temperature)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    all_trainable = trainable_backbone + list(film.parameters())
    optimizer = torch.optim.AdamW(
        all_trainable,
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_gs  = float("inf")
    patience_ctr = 0
    history = []

    print(f"\n[*] Training for up to {args.epochs} epochs "
          f"(early stop patience={args.patience})\n")

    for epoch in range(1, args.epochs + 1):
        train_stats = _run_epoch(
            active_model, oracle_model, train_loader, device,
            optimizer, loss_gs, loss_pu, loss_kl,
            args.w_gs, args.w_pu, args.w_kl, is_train=True,
        )
        val_stats = _run_epoch(
            active_model, oracle_model, val_loader, device,
            None, loss_gs, loss_pu, loss_kl,
            args.w_gs, args.w_pu, args.w_kl, is_train=False,
        )
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss={train_stats['loss']:.4f} "
            f"(gs={train_stats['gs']:.4f} pu={train_stats['pu']:.4f} kl={train_stats['kl']:.4f}) | "
            f"val gs={val_stats['gs']:.4f} | lr={lr_now:.2e}"
        )

        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})

        # ── Checkpoint + early stop ───────────────────────────────────────────
        if val_stats["gs"] < best_val_gs:
            best_val_gs  = val_stats["gs"]
            patience_ctr = 0

            ckpt_state = {
                "epoch":        epoch,
                "val_gs":       best_val_gs,
                "active_model": active_model.state_dict(),
                "film":         film.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "hparams":      active_model.hparams,
            }
            best_path = out_dir / "iceberg_ms3_best.pt"
            torch.save(ckpt_state, best_path)
            print(f"  → Best checkpoint saved (val_gs={best_val_gs:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\n[!] Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    # ── Save training history ─────────────────────────────────────────────────
    hook_handle.remove()
    history_path = out_dir / "train_history.json"
    with open(history_path, "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"\n[Done] Best val GS-BCE = {best_val_gs:.4f}")
    print(f"       Checkpoint → {out_dir / 'iceberg_ms3_best.pt'}")
    print(f"       History    → {history_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune ICEBERG-Generate (FragGNN) with empirical MS³ edges."
    )
    # Data
    p.add_argument("--feather",      required=True,
                   help="Path to ms3_edge_labels.feather")
    p.add_argument("--ckpt_path",    required=True,
                   help="Path to FragGNN PyTorch-Lightning checkpoint (.ckpt)")
    p.add_argument("--output_dir",   default="fiar_pipeline/results/iceberg_ms3",
                   help="Directory for checkpoints and history (default: %(default)s)")
    p.add_argument("--ion_mode",     default=None,
                   choices=["positive", "negative"],
                   help="Filter dataset to one polarity (default: both)")
    p.add_argument("--max_mol_size", type=int, default=80,
                   help="Maximum heavy-atom count per molecule (default: %(default)s)")
    p.add_argument("--val_frac",     type=float, default=0.10,
                   help="Fraction of data held out for validation (default: %(default)s)")
    # Training
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience",     type=int,   default=5,
                   help="Early-stopping patience in epochs (default: %(default)s)")
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--device",       default="cuda:0")
    # Loss weights
    p.add_argument("--w_gs",         type=float, default=1.0,
                   help="Weight for GaussianSoftTargetBCE (default: %(default)s)")
    p.add_argument("--w_pu",         type=float, default=0.5,
                   help="Weight for MaskedPUMarginLoss (default: %(default)s)")
    p.add_argument("--w_kl",         type=float, default=0.3,
                   help="Weight for KLDistillationLoss (default: %(default)s)")
    p.add_argument("--pu_margin",    type=float, default=0.3,
                   help="Margin γ for MaskedPUMarginLoss (default: %(default)s)")
    p.add_argument("--kl_temperature", type=float, default=2.0,
                   help="Temperature T for KLDistillationLoss (default: %(default)s)")
    return p.parse_args()


if __name__ == "__main__":
    main(_cli())
