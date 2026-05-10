"""
Phase 4 — Multi-Head Sinkhorn-FiLM Optimal Transport Probe
====================================================
Trains the SinkhornFiLMProbe (Thermodynamic Modulator + Multi-Head Wasserstein Probe)
on top of the strictly frozen Phase 2.5 MassFormer backbone.

Key Upgrades from Single-Head Phase 4:
-------------------------------------------------------
Architecture : 8-Head Sinkhorn to expand gradient bandwidth 8x
Math         : Log-Domain Sinkhorn-Knopp (eps=0.01) to prevent NaN underflow
Scheduler    : 2-Epoch Linear Warmup into Cosine Annealing to protect FiLM
Loss         : BinaryFocalLoss

Usage
-----
cd /data/nas-gpu/wang/tmach007/ms-pred

/data/nas-gpu/wang/tmach007/ms-pred/y/envs/ms-gen/bin/python \
    fiar_pipeline/train/train_phase4_probe.py \
    --config fiar_pipeline/configs/fiar_nist20.yml \
    --phase2_checkpoint /path/to/best.pt \
    [--phase4_ckpt  /path/to/existing_phase4.pt]   # optional resume
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ── Repo paths ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SSP  = Path("/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor")
_MF   = Path("/data/nas-gpu/wang/tmach007/massformer/src")

sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SSP / "model" / "fiar"))
sys.path.insert(0, str(_SSP / "model" / "transport_model" / "bifurcate"))
sys.path.insert(0, str(_MF))

# ── NAMESPACE INJECTION (Fixes NameError: 'Predictor' is not defined) ─────────
from fiar_pipeline.data_loaders.siamese_fragment_dataloader import SiameseFragmentDataset
from fiar_pipeline.model.symmetric_colbert_probe import Phase3ColBERTModel

# ── Phase 4 Sinkhorn Imports ──────────────────────────────────────────────────
from fiar_pipeline.data_loaders.phase4_dataloader import (  # noqa: E402
    SinkhornFragmentDataset,
    sinkhorn_collate_fn,
)
from fiar_pipeline.model.phase4_colbert import SinkhornFiLMProbe   # noqa: E402

# ── Optional wandb ────────────────────────────────────────────────────────────
try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False
    print("[Train] wandb not installed — metrics logged to stdout only.")


# ── Focal loss (Phase 3 Standard) ─────────────────────────────────────────────
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
    cfg         = load_config(args.config)
    data_cfg    = cfg["data"]
    train_cfg   = cfg["train"]
    iceberg_cfg = cfg["iceberg"]
    phase3_cfg  = cfg.get("phase3", {})
    phase4_cfg  = cfg.get("phase4", {})

    device = torch.device(
        f"cuda:{train_cfg['gpu_id']}" if torch.cuda.is_available() else "cpu"
    )
    print(f"[Train] Device: {device}")
    
    # ── Strict Phase 4 Output Routing ──
    out_dir = train_cfg.get("phase4_output_dir", str(Path(train_cfg["output_dir"]).parent / "phase4_probe"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Train] Output Directory: {out_dir}")

    if _WANDB_OK:
        wandb.init(
            project=train_cfg.get("wandb_project", "fiar-iceberg-phase4").replace("phase3", "phase4"),
            config={**iceberg_cfg, **train_cfg, **phase3_cfg, **phase4_cfg},
        )

    # ── Datasets ──────────────────────────────────────────────────────────────
    fragment_cache = args.fragment_cache_path or iceberg_cfg.get("fragment_cache_path")
    max_k = phase4_cfg.get("max_fragments", phase3_cfg.get("max_fragments", 50))

    common_kwargs = dict(
        spec_df_path        = data_cfg["spec_df"],
        mol_df_path         = data_cfg["mol_df"],
        phase2_loader_dir   = data_cfg["phase2_loader_dir"],
        fragment_cache_path = fragment_cache,
        max_k               = max_k,
        morgan_nbits        = iceberg_cfg.get("morgan_nbits", 2048),
    )

    primary = SinkhornFragmentDataset(
        feather_path = data_cfg["pairs_train"],
        graphs_path  = data_cfg["graphs"],
        **common_kwargs,
    )

    if data_cfg.get("pairs_val"):
        train_ds = primary
        val_ds   = SinkhornFragmentDataset(
            feather_path = data_cfg["pairs_val"],
            graphs_path  = data_cfg["graphs"],
            **common_kwargs,
        )
    else:
        n_train = int(0.8 * len(primary))
        train_ds, val_ds = random_split(primary, [n_train, len(primary) - n_train])

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True,  collate_fn=sinkhorn_collate_fn,
        num_workers=6, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,   batch_size=train_cfg["batch_size"],
        shuffle=False, collate_fn=sinkhorn_collate_fn,
        num_workers=6, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    phase2_ckpt = args.phase2_checkpoint or train_cfg["phase2_checkpoint"]
    
    s_eps   = phase4_cfg.get("sinkhorn_eps", 0.01)
    s_iters = phase4_cfg.get("sinkhorn_iters", 20)
    n_heads = phase4_cfg.get("num_heads", 8)
    rho_val = phase4_cfg.get("rho_init", 1.0)
    tau_val = phase4_cfg.get("tau_init", 5.0)

    model = SinkhornFiLMProbe(
        cfg=cfg,
        phase2_ckpt=phase2_ckpt,
        device=device,
        max_fragments=max_k,
        sinkhorn_eps=s_eps,
        sinkhorn_iters=s_iters,
        num_heads=n_heads,
        load_backbone=True,
        rho_init=rho_val,
        tau_init=tau_val
    ).to(device)

    if args.phase4_ckpt and Path(args.phase4_ckpt).exists():
        print(f"[Train] Resuming Phase 4 from: {args.phase4_ckpt}")
        p4_state = torch.load(args.phase4_ckpt, map_location=device, weights_only=False)
        if isinstance(p4_state, dict) and "state_dict" in p4_state:
            p4_state = p4_state["state_dict"]
        model.load_state_dict(p4_state, strict=False)

    # ── Verify freeze & Parameter Audit ───────────────────────────────────────
    frozen_leak = [name for name, p in model.backbone.named_parameters() if p.requires_grad]
    if frozen_leak:
        for p in model.backbone.parameters():
            p.requires_grad = False

    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[Train] Trainable params ({n_heads}-Head Sinkhorn-FiLM): {n_trainable:,}")

    # ── Optimizer / scheduler / loss ──────────────────────────────────────────
    # Defaulting to 3e-3 as requested by the DL expert
    base_lr = train_cfg.get("learning_rate", 3e-3)
    optimizer = AdamW(trainable_params, lr=base_lr, weight_decay=1e-3)
    
    # Linear Warmup (First 2 epochs) into Cosine Annealing
    epochs = train_cfg["epochs"]
    warmup_epochs = 2
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-5)
    
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )
    
    criterion = BinaryFocalLoss(
        alpha = train_cfg["focal_alpha"],
        gamma = train_cfg["focal_gamma"]
    )

    best_auc  = 0.0
    best_ckpt = Path(out_dir) / "fiar_phase4_sinkhorn_best.pt"

    for epoch in range(1, epochs + 1):
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # ── BatchNorm patch ───────────────────────────────────────────────────
        model.eval()
        model.film_mlp.train()
        model.wasserstein_probe.train()

        t_loss = 0.0
        bar = tqdm(train_loader, desc=f"E{epoch}/{epochs} [LR: {current_lr:.5f}]")

        for batch_A, batch_B, _t_sim, t_lbl in bar:
            if batch_A is None:
                continue

            def _to_device(d: dict) -> dict:
                return {
                    k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in d.items()
                }

            batch_A = _to_device(batch_A)
            batch_B = _to_device(batch_B)
            t_lbl   = t_lbl.to(device, non_blocking=True).float()

            optimizer.zero_grad()

            logits, w_dist = model(batch_A, batch_B)
            logits  = logits.view(-1)
            labels  = t_lbl.view(-1)

            loss = criterion(logits, labels)
            if torch.isnan(loss):
                print("[Train] NaN loss — halting.")
                sys.exit(1)

            loss.backward()
            
            # Clip gradients to ensure Sinkhorn stability
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            optimizer.step()
            t_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        v_loss     = 0.0
        all_logits: list = []
        all_labels: list = []

        with torch.no_grad():
            for batch_A, batch_B, _t_sim, t_lbl in val_loader:
                if batch_A is None:
                    continue

                batch_A = _to_device(batch_A)
                batch_B = _to_device(batch_B)
                t_lbl   = t_lbl.to(device, non_blocking=True).float()

                logits, w_dist = model(batch_A, batch_B)
                logits  = logits.view(-1)
                labels  = t_lbl.view(-1)

                v_loss += criterion(logits, labels).item()
                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        n_val = max(len(val_loader), 1)
        try:
            auc = roc_auc_score(all_labels, all_logits)
        except ValueError:
            auc = 0.0

        current_tau = model.tau.item()
        # Compute the active rho using softplus (matching the forward pass)
        import torch.nn.functional as F
        current_rho = F.softplus(model.rho_raw).item()

        print(f"\n{'─'*70}")
        print(f" Epoch {epoch:3d}  "
              f"Train: {t_loss/max(len(train_loader),1):.4f}  "
              f"Val: {v_loss/n_val:.4f}  "
              f"AUC: {auc:.4f}  "
              f"Tau: {current_tau:.3f}  "
              f"Rho: {current_rho:.3f}")
        print(f"{'─'*70}")

        if _WANDB_OK:
            wandb.log(dict(
                epoch      = epoch,
                train_loss = t_loss / max(len(train_loader), 1),
                val_loss   = v_loss / n_val,
                val_auc    = auc,
                lr         = current_lr,
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
        description="Phase 4: train Multi-Head Sinkhorn-FiLM Optimal Transport probe."
    )
    p.add_argument("--config", default="fiar_pipeline/configs/fiar_nist20.yml",
                   help="Path to YAML config (default: fiar_nist20.yml)")
    p.add_argument("--phase2_checkpoint", default=None,
                   help="Override train.phase2_checkpoint in config")
    p.add_argument("--phase4_ckpt", default=None,
                   help="Optional: resume from a previous Phase 4 checkpoint")
    p.add_argument("--fragment_cache_path", default=None,
                   help="Override iceberg.fragment_cache_path in config")
    return p.parse_args()


if __name__ == "__main__":
    train(_cli())