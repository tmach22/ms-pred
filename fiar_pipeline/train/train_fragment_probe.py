"""
Phase 3 — Mass-Weighted Symmetric ColBERT Probe
================================================
Trains only the SymmetricColBERTHead (DynamicWeightingMLP + binary_head) on
top of the strictly frozen Phase 2.5 MassFormer backbone.

Design constraints
------------------
Freeze:      MassFormer backbone + Phase 2.5 Thermodynamic Adapters are fully
             frozen (requires_grad=False). Only SymmetricColBERTHead trains.
Loss:        Binary Focal Loss (gamma=3.0, alpha=0.75). Symmetric MaxSim
             mathematically compresses the score range, so high gamma is needed
             to rescue the gradient signal for true analog pairs.
Optimizer:   AdamW lr=1e-3 (aggressive — backbone is frozen so no catastrophic
             forgetting risk).
BN patch:    model.eval() globally at epoch start; model.colbert_head.train()
             to thaw only the head. Prevents frozen BatchNorm layers from
             updating running statistics and corrupting the learned manifold.
Namespace:   Phase 2.5 checkpoint loaded with strict=False; missing_keys and
             unexpected_keys are printed at startup for explicit verification.
Data:        Top-50 ICEBERG fragments per molecule from fragment_cache.pt.
             Metadata tensor: [relative_mass, log(prob_gen+eps), NCE=CE/50].

Usage
-----
cd /data/nas-gpu/wang/tmach007/ms-pred

/data/nas-gpu/wang/tmach007/ms-pred/y/envs/ms-gen/bin/python \\
    fiar_pipeline/train/train_fragment_probe.py \\
    --config fiar_pipeline/configs/fiar_nist20.yml \\
    --phase2_checkpoint /path/to/best.pt \\
    --phase3_ckpt       /path/to/existing_phase3.pt  # optional resume
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ── Repo paths ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SSP  = Path("/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor")
_MF   = Path("/data/nas-gpu/wang/tmach007/massformer/src")

sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

# SpectralSimilarityPredictor model directories
sys.path.insert(0, str(_SSP / "model" / "fiar"))
sys.path.insert(0, str(_SSP / "model" / "transport_model" / "bifurcate"))

# MassFormer package
sys.path.insert(0, str(_MF))

from fiar_pipeline.data_loaders.siamese_fragment_dataloader import (  # noqa: E402
    SiameseFragmentDataset,
    siamese_frag_collate_fn,
)
from fiar_pipeline.model.symmetric_colbert_probe import Phase3ColBERTModel  # noqa: E402

# ── Optional wandb ────────────────────────────────────────────────────────────
try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False
    print("[Train] wandb not installed — metrics logged to stdout only.")


# ── Focal loss ────────────────────────────────────────────────────────────────

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self._bce  = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self._bce(logits, targets)
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    cfg        = load_config(args.config)
    data_cfg   = cfg["data"]
    train_cfg  = cfg["train"]
    iceberg_cfg = cfg["iceberg"]
    phase3_cfg = cfg.get("phase3", {})

    device = torch.device(
        f"cuda:{train_cfg['gpu_id']}" if torch.cuda.is_available() else "cpu"
    )
    print(f"[Train] Device: {device}")
    os.makedirs(train_cfg["output_dir"], exist_ok=True)

    if _WANDB_OK:
        wandb.init(
            project=train_cfg["wandb_project"],
            config={**iceberg_cfg, **train_cfg, **phase3_cfg},
        )

    # ── Datasets ──────────────────────────────────────────────────────────────
    fragment_cache = (
        args.fragment_cache_path or iceberg_cfg.get("fragment_cache_path")
    )
    # Top-50 fragments fed to the ColBERT head
    max_k = phase3_cfg.get("max_fragments", 50)

    common_kwargs = dict(
        spec_df_path        = data_cfg["spec_df"],
        mol_df_path         = data_cfg["mol_df"],
        phase2_loader_dir   = data_cfg["phase2_loader_dir"],
        fragment_cache_path = fragment_cache,
        max_k               = max_k,
        morgan_nbits        = iceberg_cfg.get("morgan_nbits", 2048),
    )

    primary = SiameseFragmentDataset(
        feather_path = data_cfg["pairs_train"],
        graphs_path  = data_cfg["graphs"],
        **common_kwargs,
    )

    if data_cfg.get("pairs_val"):
        train_ds = primary
        val_ds   = SiameseFragmentDataset(
            feather_path = data_cfg["pairs_val"],
            graphs_path  = data_cfg["graphs"],
            **common_kwargs,
        )
    else:
        n_train = int(0.8 * len(primary))
        train_ds, val_ds = random_split(
            primary, [n_train, len(primary) - n_train]
        )

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True,  collate_fn=siamese_frag_collate_fn,
        num_workers=6, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,   batch_size=train_cfg["batch_size"],
        shuffle=False, collate_fn=siamese_frag_collate_fn,
        num_workers=6, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # Phase 2.5 checkpoint path (CLI overrides config)
    phase2_ckpt = args.phase2_checkpoint or train_cfg["phase2_checkpoint"]

    # Phase3ColBERTModel constructor:
    #   1. Builds the frozen Phase 2.5 backbone
    #   2. Runs the strict=False namespace check and prints missing/unexpected keys
    #   3. Adds the trainable SymmetricColBERTHead
    model = Phase3ColBERTModel(
        cfg=cfg,
        phase2_ckpt=phase2_ckpt,
        device=device,
        max_fragments=max_k,
    ).to(device)

    # Optional Phase 3 resume (e.g. interrupted run)
    if args.phase3_ckpt and Path(args.phase3_ckpt).exists():
        print(f"[Train] Resuming Phase 3 from: {args.phase3_ckpt}")
        p3_state = torch.load(args.phase3_ckpt, map_location=device,
                               weights_only=False)
        if isinstance(p3_state, dict) and "state_dict" in p3_state:
            p3_state = p3_state["state_dict"]
        model.load_state_dict(p3_state, strict=False)

    # ── Verify freeze: only colbert_head must have requires_grad=True ─────────
    trainable_params = [p for p in model.colbert_head.parameters()]
    frozen_leak = [
        name for name, p in model.backbone.named_parameters() if p.requires_grad
    ]
    if frozen_leak:
        print(f"[Train] WARNING: {len(frozen_leak)} backbone params have "
              f"requires_grad=True — forcing freeze.")
        for p in model.backbone.parameters():
            p.requires_grad = False
        trainable_params = [p for p in model.colbert_head.parameters()]

    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[Train] Trainable params (ColBERT head only): {n_trainable:,}")

    # ── Optimizer / scheduler / loss ──────────────────────────────────────────
    # Aggressive lr=1e-3: safe because the backbone is completely frozen.
    optimizer = AdamW(trainable_params, lr=train_cfg["learning_rate"],
                      weight_decay=1e-3)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"], eta_min=1e-5
    )
    criterion = BinaryFocalLoss(
        alpha=train_cfg["focal_alpha"],
        gamma=train_cfg["focal_gamma"],
    )

    best_auc  = 0.0
    best_ckpt = Path(train_cfg["output_dir"]) / "fiar_phase3_colbert_best.pt"

    for epoch in range(1, train_cfg["epochs"] + 1):

        # ── BatchNorm patch ───────────────────────────────────────────────────
        # Step 1: freeze ALL layers (including BatchNorm running stats).
        model.eval()
        # Step 2: thaw ONLY the ColBERT head so its weights receive gradients.
        model.colbert_head.train()

        t_loss = 0.0
        bar = tqdm(train_loader, desc=f"E{epoch}/{train_cfg['epochs']} [train]")

        for batch_A, batch_B, _t_sim, t_lbl in bar:
            if batch_A is None:
                continue

            batch_A = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch_A.items()
            }
            batch_B = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch_B.items()
            }
            t_lbl = t_lbl.to(device, non_blocking=True).float()

            optimizer.zero_grad()

            logits = model(batch_A, batch_B).view(-1)   # [batch]
            labels = t_lbl.view(-1)

            loss = criterion(logits, labels)
            if torch.isnan(loss):
                print("[Train] NaN focal loss — halting.")
                sys.exit(1)

            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            bar.set_postfix(focal=f"{loss.item():.4f}")

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        v_loss     = 0.0
        all_logits: list = []
        all_labels: list = []

        with torch.no_grad():
            for batch_A, batch_B, _t_sim, t_lbl in val_loader:
                if batch_A is None:
                    continue

                batch_A = {
                    k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in batch_A.items()
                }
                batch_B = {
                    k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in batch_B.items()
                }
                t_lbl = t_lbl.to(device, non_blocking=True).float()

                logits = model(batch_A, batch_B).view(-1)
                labels = t_lbl.view(-1)

                v_loss += criterion(logits, labels).item()
                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        n_val = max(len(val_loader), 1)
        try:
            auc = roc_auc_score(all_labels, all_logits)
        except ValueError:
            auc = 0.0

        print(f"\n{'─'*56}")
        print(f" Epoch {epoch:3d}  "
              f"Train Focal: {t_loss/max(len(train_loader),1):.4f}  "
              f"Val Focal: {v_loss/n_val:.4f}  "
              f"ROC-AUC: {auc:.4f}")
        print(f"{'─'*56}")

        if _WANDB_OK:
            wandb.log(dict(
                epoch       = epoch,
                train_focal = t_loss / max(len(train_loader), 1),
                val_focal   = v_loss / n_val,
                val_auc     = auc,
            ))

        scheduler.step()

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_ckpt)
            print(f"[Train] ★ Best AUC {best_auc:.4f} → {best_ckpt}")

    if _WANDB_OK:
        wandb.finish()
    print(f"\n[Train] Done. Best ROC-AUC: {best_auc:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3: train Mass-Weighted Symmetric ColBERT probe."
    )
    p.add_argument("--config", default="fiar_pipeline/configs/fiar_nist20.yml",
                   help="Path to YAML config (default: fiar_nist20.yml)")
    p.add_argument("--phase2_checkpoint", default=None,
                   help="Override train.phase2_checkpoint in config")
    p.add_argument("--phase3_ckpt", default=None,
                   help="Optional: resume from a previous Phase 3 checkpoint")
    p.add_argument("--fragment_cache_path", default=None,
                   help="Override iceberg.fragment_cache_path in config")
    return p.parse_args()


if __name__ == "__main__":
    train(_cli())
